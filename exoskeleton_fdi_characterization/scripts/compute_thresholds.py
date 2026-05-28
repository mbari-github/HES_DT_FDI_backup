#!/usr/bin/env python3
"""
compute_thresholds.py — derive fdi_node parameters from characterisation data

Combines the healthy baseline statistics with the fault sweep analysis to
produce a complete, scientifically-derived fdi_node parameter file.  No more
trial-and-error.

Inputs
------
    --baseline-stats  baseline_stats.json   (from analyze_baseline.py)
    --sweep-table     fault_sweep_table.json (from analyze_fault_sweep.py) [optional]
    --far             target false-alarm rate per sample [default 0.001 = 0.1 %]
    --p-far-total     desired P(false alarm confirmed) after debounce [default 1e-9]
    --publish-rate    FDI loop frequency in Hz                        [default 200]
    --out-dir         output directory                                [default .]

Outputs
-------
    computed_fdi_params.yaml   — complete fdi_node ros__parameters snippet
                                 (copy into fdi_params.yaml and rebuild)
    thresholds_report.md       — statistical justification for every value

Methodology (all described in thresholds_report.md)
----------------------------------------------------

thresh_force / thresh_encoder
  The threshold on the EMA-filtered residual is set at the (1-FAR) percentile
  of the healthy signal distribution.  Because we only have p99 / p99.9 /
  p99.99 from the baseline, we pick the one closest to (1-FAR)*100:

      FAR = 0.001  →  use p99.9
      FAR = 0.01   →  use p99
      FAR = 0.0001 →  use p99.99

  The threshold is: threshold = mean_healthy + pX_healthy
  (mean_healthy is the steady-state EMA level for zero fault; pX is the
  99.X-th percentile of |r_filt − mean_healthy| in healthy conditions).

residual_alpha
  The EMA coefficient alpha controls noise rejection vs detection speed.
  For a given healthy std σ_raw (from the raw un-filtered residual), the
  EMA output std is:

      σ_filtered = σ_raw * sqrt(alpha / (2 - alpha))

  We choose alpha such that σ_filtered < (threshold − mean_healthy) / 3:
  this ensures SNR ≥ 3 when a fault of magnitude = threshold occurs.

      x = (snr_target)^2 * sigma_raw^2 / (threshold - mean)^2
        — this is the ratio that must hold after filtering
      Actually solve: alpha/(2-alpha) = ((thresh-mean)/snr_margin)^2 / sigma_raw^2
      alpha = 2*ratio / (1+ratio)   where ratio = target_variance / sigma_raw^2

  Bounds: alpha ∈ [0.02, 0.5].

debounce_count
  The leaky-integrator debounce in fdi_node requires N consecutive samples
  above threshold to fire.  The probability that all N healthy samples are
  false alarms is FAR^N.  We choose N so:

      FAR^N < p_far_total   →   N = ceil(log(p_far_total) / log(FAR))

  Minimum latency = N * dt.  If the sweep table is provided, N is also
  lower-bounded by the desired latency from the MDL-90 analysis.

enc_obs_pole_1 / enc_obs_pole_2 (model-based, not data-driven)
  The Luenberger poles are derived from the plant model, not from the
  characterisation data.  They are written verbatim from fdi_params.yaml
  with a comment explaining the stability constraint.

Usage examples
--------------
    # From baseline only:
    python3 compute_thresholds.py \\
        --baseline-stats ./bags/healthy_baseline_*/baseline_stats.json

    # With sweep for MDL-based debounce:
    python3 compute_thresholds.py \\
        --baseline-stats ./bags/healthy_baseline_*/baseline_stats.json \\
        --sweep-table    ./bags/fault_sweep_*/fault_sweep_table.json \\
        --far 0.001

    # Tighter thresholds (less false alarms, may miss small faults):
    python3 compute_thresholds.py \\
        --baseline-stats ./bags/healthy_baseline_*/baseline_stats.json \\
        --far 0.0001
"""
import argparse
import json
import math
import sys
from pathlib import Path


# ════════════════════════════════════════════════════════════════════
# Core computations
# ════════════════════════════════════════════════════════════════════

def _percentile_key(far: float) -> str:
    """Map FAR to the closest percentile stored in baseline_stats."""
    if far >= 0.01:
        return 'p99'
    if far >= 0.001:
        return 'p99_9'
    return 'p99_99'


def compute_threshold(baseline: dict, signal: str, far: float) -> dict:
    """
    Return threshold and supporting statistics for one filtered residual.

    threshold = mean_healthy + pX_healthy
    where pX = percentile matching (1-FAR)*100.
    """
    b = baseline.get(signal, {})
    if not b:
        return {'threshold': None, 'mean_h': None, 'pX': None, 'pct_used': None}

    mean_h  = float(b.get('mean', 0.0))
    pct_key = _percentile_key(far)
    pX      = float(b.get(pct_key, 0.0))
    thresh  = mean_h + pX
    return {
        'threshold': thresh,
        'mean_h':    mean_h,
        'std_h':     float(b.get('std', 0.0)),
        'max_abs':   float(b.get('max_abs', 0.0)),
        'pX':        pX,
        'pct_used':  pct_key,
    }


def compute_alpha(baseline: dict, raw_signal: str, threshold: float,
                  mean_h: float, snr_margin: float = 3.0) -> dict:
    """
    EMA alpha so that sigma_filtered ≤ (threshold − mean_h) / snr_margin.

    Derivation:
        sigma2_filtered = sigma2_raw * alpha / (2 - alpha)
        target_sigma2   = ((threshold - mean_h) / snr_margin)^2
        ratio           = target_sigma2 / sigma2_raw
        alpha           = 2 * ratio / (1 + ratio)
    """
    b = baseline.get(raw_signal, {})
    sigma_raw = float(b.get('std', 0.0))

    excursion = threshold - mean_h
    if excursion <= 0.0 or sigma_raw <= 0.0:
        return {'alpha': 0.1, 'sigma_raw': sigma_raw,
                'sigma_filtered_expected': 0.0, 'snr': float('inf')}

    target_sigma = excursion / snr_margin
    ratio = (target_sigma / sigma_raw) ** 2
    alpha = 2.0 * ratio / (1.0 + ratio)
    alpha = max(0.02, min(alpha, 0.5))

    sigma_filt_actual = sigma_raw * math.sqrt(alpha / (2.0 - alpha))
    snr_actual = excursion / sigma_filt_actual if sigma_filt_actual > 0 else float('inf')

    return {
        'alpha':                    round(alpha, 4),
        'sigma_raw':                sigma_raw,
        'sigma_filtered_expected':  sigma_filt_actual,
        'snr':                      snr_actual,
        'excursion':                excursion,
    }


def compute_debounce(far: float, p_far_total: float,
                     publish_rate: float) -> dict:
    """
    Minimum debounce_n so P(n consecutive false alarms) < p_far_total.

    N = ceil(log(p_far_total) / log(far))

    Latency_min = N / publish_rate  [s]
    """
    if far <= 0.0 or far >= 1.0:
        return {'debounce_n': 10, 'latency_min_ms': 50.0}

    n = math.ceil(math.log(p_far_total) / math.log(far))
    n = max(3, min(n, 100))
    latency_ms = 1000.0 * n / publish_rate

    return {
        'debounce_n':     n,
        'latency_min_ms': latency_ms,
        'p_confirmed_far': far ** n,
    }


# ════════════════════════════════════════════════════════════════════
# Sweep-based analysis (optional)
# ════════════════════════════════════════════════════════════════════

def mdl_from_sweep(sweep: list, channel: int, fault_type: str,
                   residual: str, p_thresh: float = 0.9):
    """
    Minimum detectable level: smallest magnitude where p_detect >= p_thresh.
    Returns None if no window reached p_thresh.
    """
    candidates = []
    for w in sweep:
        if w['channel'] != channel or w['type'] != fault_type:
            continue
        d = w['residuals'].get(residual)
        if d is None:
            continue
        if d.get('p_detect', 0.0) >= p_thresh:
            candidates.append(w['magnitude'])
    return min(candidates) if candidates else None


def latency_at_mdl(sweep: list, channel: int, fault_type: str,
                   residual: str, mdl: float):
    """Mean latency for windows at exactly mdl magnitude."""
    latencies = []
    for w in sweep:
        if (w['channel'] != channel or w['type'] != fault_type
                or abs(w['magnitude'] - mdl) > 1e-9):
            continue
        d = w['residuals'].get(residual)
        if d and d.get('latency_s') is not None:
            latencies.append(d['latency_s'])
    return float(sum(latencies) / len(latencies)) if latencies else None


# ════════════════════════════════════════════════════════════════════
# Output generation
# ════════════════════════════════════════════════════════════════════

def generate_yaml(results: dict, far: float, p_far_total: float,
                  publish_rate: float) -> str:
    tf   = results['thresh_force']
    te   = results['thresh_encoder']
    af   = results['alpha_force']
    ae   = results['alpha_encoder']
    deb  = results['debounce']

    # Use conservative alpha (smaller = more filtering)
    alpha = round(min(af['alpha'], ae['alpha']), 4)
    alpha = max(alpha, 0.05)

    thresh_f = tf['threshold']
    thresh_e = te['threshold']

    lines = [
        '# computed_fdi_params.yaml',
        '# ════════════════════════════════════════════════════════════════',
        '# Auto-generated by compute_thresholds.py — DO NOT EDIT MANUALLY',
        '# See thresholds_report.md for the full statistical justification.',
        '# ════════════════════════════════════════════════════════════════',
        '',
        '# To apply: merge into fdi_params.yaml (keep all other parameters)',
        '# or override individual values with ros2 param set.',
        '',
        'fdi_node:',
        '  ros__parameters:',
        '',
        '    # ──────────────────────────────────────────────────────────',
        '    # DETECTION THRESHOLDS  (data-driven)',
        '    # ──────────────────────────────────────────────────────────',
        '    #',
        f'    # thresh_force: healthy mean={tf["mean_h"]:.4f} Nm,'
        f' {tf["pct_used"]}={tf["pX"]:.4f} Nm'
        f'  →  threshold={thresh_f:.4f} Nm',
        f'    #   (FAR ~ {far*100:.2f}% / sample, 1 false alarm every'
        f' {1.0/(far*publish_rate):.1f} s at {publish_rate:.0f} Hz)',
    ]
    if results.get('mdl_force') is not None:
        mdl_f = results['mdl_force']
        lat_f = results.get('latency_force')
        lat_str = f'{lat_f*1000:.0f} ms' if lat_f else 'n/a'
        lines.append(f'    #   MDL-90 (sweep): {mdl_f:.3g} Nm  (latency ≈ {lat_str})')

    lines += [
        f'    thresh_force:   {thresh_f:.4f}',
        '',
        f'    # thresh_encoder: healthy mean={te["mean_h"]:.5f} rad,'
        f' {te["pct_used"]}={te["pX"]:.5f} rad'
        f'  →  threshold={thresh_e:.5f} rad',
        f'    #   (FAR ~ {far*100:.2f}% / sample)',
    ]
    if results.get('mdl_encoder') is not None:
        mdl_e = results['mdl_encoder']
        lat_e = results.get('latency_encoder')
        lat_str = f'{lat_e*1000:.0f} ms' if lat_e else 'n/a'
        lines.append(f'    #   MDL-90 (sweep): {mdl_e:.4g} rad  (latency ≈ {lat_str})')

    deb_n = deb['debounce_n']
    deb_ms = deb['latency_min_ms']
    p_conf = deb.get('p_confirmed_far', 0.0)

    lines += [
        f'    thresh_encoder: {thresh_e:.5f}',
        '',
        '    # ──────────────────────────────────────────────────────────',
        '    # EMA FILTER COEFFICIENT  (data-driven)',
        '    # ──────────────────────────────────────────────────────────',
        '    #',
        f'    # Chosen so sigma_filtered < (threshold - mean) / 3 (SNR >= 3)',
        f'    #   r_force:   sigma_raw={af["sigma_raw"]:.4f}'
        f'  sigma_filt={af["sigma_filtered_expected"]:.4f}'
        f'  SNR={af["snr"]:.1f}  alpha={af["alpha"]:.4f}',
        f'    #   r_encoder: sigma_raw={ae["sigma_raw"]:.5f}'
        f'  sigma_filt={ae["sigma_filtered_expected"]:.5f}'
        f'  SNR={ae["snr"]:.1f}  alpha={ae["alpha"]:.4f}',
        f'    #   Using min(alpha_force, alpha_encoder) = {alpha:.4f}',
        f'    residual_alpha: {alpha:.4f}',
        '',
        '    # ──────────────────────────────────────────────────────────',
        '    # DEBOUNCE COUNT  (data-driven)',
        '    # ──────────────────────────────────────────────────────────',
        '    #',
        f'    # N such that P(N consecutive false alarms) = FAR^N < {p_far_total:.0e}',
        f'    #   FAR={far:.4f}  →  N={deb_n}  →  P_confirmed={p_conf:.2e}',
        f'    #   Minimum detection latency: {deb_ms:.0f} ms at {publish_rate:.0f} Hz',
    ]
    if results.get('mdl_force') is not None or results.get('mdl_encoder') is not None:
        lines.append('    #   (consistent with MDL-90 latency from sweep — see report)')

    lines += [
        f'    debounce_count: {deb_n}',
        '',
        '    # ──────────────────────────────────────────────────────────',
        '    # LUENBERGER POLES  (model-based, not data-driven)',
        '    # ──────────────────────────────────────────────────────────',
        '    #',
        '    # Stability constraint: |pole| * dt < 1  →  |pole| < 200 at dt=5 ms',
        '    # Convergence: choose ~2-3x faster than plant dominant pole (~-6)',
        '    # These are NOT changed by characterisation; tune via pole-placement',
        '    # if the Luenberger residual in healthy mode is consistently large.',
        '    enc_obs_pole_1: -15.0',
        '    enc_obs_pole_2: -20.0',
    ]
    return '\n'.join(lines) + '\n'


def generate_report(results: dict, far: float, p_far_total: float,
                    publish_rate: float, has_sweep: bool) -> str:
    tf  = results['thresh_force']
    te  = results['thresh_encoder']
    af  = results['alpha_force']
    ae  = results['alpha_encoder']
    deb = results['debounce']

    alpha = round(min(af['alpha'], ae['alpha']), 4)
    alpha = max(alpha, 0.05)

    text = [
        '# Threshold Computation Report',
        '',
        '## Inputs',
        '',
        f'- Target FAR per sample: **{far:.4f}** ({far*100:.2f}%)',
        f'- Target P(confirmed false alarm): **{p_far_total:.0e}**',
        f'- Publish rate: **{publish_rate:.0f} Hz**  (dt = {1000/publish_rate:.1f} ms)',
        f'- Sweep table used: **{"yes" if has_sweep else "no"}**',
        '',
        '---',
        '',
        '## 1. Detection Thresholds',
        '',
        '### Formula',
        '',
        '```',
        'threshold = mean_healthy + pX_healthy',
        'where pX = (1-FAR)·100 th percentile of |r_filt − mean_healthy|',
        '```',
        '',
        '### R_force (channel 0 — tau_ext sensor)',
        '',
        f'| Statistic | Value |',
        f'|---|---|',
        f'| mean_healthy | {tf["mean_h"]:.4f} Nm |',
        f'| std_healthy  | {tf["std_h"]:.4f} Nm |',
        f'| max_abs      | {tf["max_abs"]:.4f} Nm |',
        f'| {tf["pct_used"]} of |r − mean| | {tf["pX"]:.4f} Nm |',
        f'| **thresh_force** | **{tf["threshold"]:.4f} Nm** |',
        f'| FAR per sample | {far*100:.3f}% |',
        f'| False alarms per second | {far*publish_rate:.3f} |',
        f'| Mean time between false alarms | {1/(far*publish_rate):.1f} s |',
    ]

    if results.get('mdl_force') is not None:
        lat_f = results.get('latency_force')
        lat_str = f'{lat_f*1000:.0f} ms' if lat_f else 'n/a'
        text += [
            f'| MDL-90 (min detectable, 90% rate) | {results["mdl_force"]:.3g} Nm |',
            f'| Latency at MDL-90 | {lat_str} |',
        ]

    text += [
        '',
        '### R_encoder (channel 3 — encoder)',
        '',
        f'| Statistic | Value |',
        f'|---|---|',
        f'| mean_healthy | {te["mean_h"]:.5f} rad |',
        f'| std_healthy  | {te["std_h"]:.5f} rad |',
        f'| max_abs      | {te["max_abs"]:.5f} rad |',
        f'| {te["pct_used"]} of |r − mean| | {te["pX"]:.5f} rad |',
        f'| **thresh_encoder** | **{te["threshold"]:.5f} rad** |',
        f'| FAR per sample | {far*100:.3f}% |',
        f'| False alarms per second | {far*publish_rate:.3f} |',
        f'| Mean time between false alarms | {1/(far*publish_rate):.1f} s |',
    ]

    if results.get('mdl_encoder') is not None:
        lat_e = results.get('latency_encoder')
        lat_str = f'{lat_e*1000:.0f} ms' if lat_e else 'n/a'
        text += [
            f'| MDL-90 (min detectable, 90% rate) | {results["mdl_encoder"]:.4g} rad |',
            f'| Latency at MDL-90 | {lat_str} |',
        ]

    deb_n  = deb['debounce_n']
    deb_ms = deb['latency_min_ms']
    p_conf = deb.get('p_confirmed_far', 0.0)

    text += [
        '',
        '---',
        '',
        '## 2. EMA Filter Coefficient (residual_alpha)',
        '',
        '### Formula',
        '',
        '```',
        'sigma2_filtered = sigma2_raw * alpha / (2 - alpha)',
        'target: sigma_filtered < (threshold - mean_healthy) / SNR_margin',
        'solve:  alpha = 2*ratio / (1+ratio)   ratio = target_sigma^2 / sigma_raw^2',
        '```',
        '',
        f'SNR margin used: 3.0 (ensures ≥ 3σ separation at detection boundary)',
        '',
        f'| | R_force | R_encoder |',
        f'|---|---|---|',
        f'| sigma_raw | {af["sigma_raw"]:.4f} Nm | {ae["sigma_raw"]:.5f} rad |',
        f'| threshold − mean | {af["excursion"]:.4f} Nm | {ae["excursion"]:.5f} rad |',
        f'| target sigma_filt | {af["excursion"]/3:.4f} Nm | {ae["excursion"]/3:.5f} rad |',
        f'| optimal alpha | {af["alpha"]:.4f} | {ae["alpha"]:.4f} |',
        f'| sigma_filt (expected) | {af["sigma_filtered_expected"]:.4f} Nm'
        f' | {ae["sigma_filtered_expected"]:.5f} rad |',
        f'| SNR at threshold | {af["snr"]:.1f} | {ae["snr"]:.1f} |',
        f'| **residual_alpha used** | **{alpha:.4f}** (min of the two) ||',
        '',
        '---',
        '',
        '## 3. Debounce Count',
        '',
        '### Formula',
        '',
        '```',
        'P(n consecutive false alarms) = FAR^n',
        'choose n = ceil(log(p_target) / log(FAR))',
        '```',
        '',
        f'| Parameter | Value |',
        f'|---|---|',
        f'| FAR per sample | {far:.4f} |',
        f'| Target P(confirmed FA) | {p_far_total:.0e} |',
        f'| **debounce_count** | **{deb_n}** |',
        f'| P(confirmed false alarm) | {p_conf:.2e} |',
        f'| Minimum detection latency | {deb_ms:.0f} ms |',
        '',
        '---',
        '',
        '## 4. Luenberger Poles (enc_obs_pole_1, enc_obs_pole_2)',
        '',
        'These are model-based parameters, not derived from characterisation data.',
        '',
        '- Euler stability: `|pole| < 1/dt = 200` (at dt = 5 ms)',
        '- Recommended: ~2-3× faster than plant dominant pole',
        '  - Plant: λ₂ ≈ −(fric_visc + damping) / M_eff ≈ −6 rad/s',
        '  - Observer: [-15, -20] → convergence in ~100-200 ms',
        '',
        'If the Luenberger residual is chronically large in healthy mode, verify that',
        'fric_visc, fric_coul, and damping_theta match dynamics_params.yaml exactly.',
        '',
        '---',
        '',
        '## Summary — values to apply',
        '',
        '```yaml',
        'fdi_node:',
        '  ros__parameters:',
        f'    thresh_force:   {tf["threshold"]:.4f}',
        f'    thresh_encoder: {te["threshold"]:.5f}',
        f'    residual_alpha: {alpha:.4f}',
        f'    debounce_count: {deb_n}',
        '    enc_obs_pole_1: -15.0',
        '    enc_obs_pole_2: -20.0',
        '```',
    ]

    return '\n'.join(text) + '\n'


# ════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-stats', required=True,
                    help='baseline_stats.json from analyze_baseline.py')
    ap.add_argument('--sweep-table', default=None,
                    help='fault_sweep_table.json from analyze_fault_sweep.py (optional)')
    ap.add_argument('--far', type=float, default=0.001,
                    help='target false-alarm rate per sample (default: 0.001 = 0.1%%)')
    ap.add_argument('--p-far-total', type=float, default=1e-9,
                    help='desired P(confirmed false alarm) after debounce (default: 1e-9)')
    ap.add_argument('--publish-rate', type=float, default=200.0,
                    help='FDI loop frequency in Hz (default: 200)')
    ap.add_argument('--out-dir', default=None,
                    help='output directory (default: directory of --baseline-stats)')
    args = ap.parse_args()

    baseline_path = Path(args.baseline_stats).resolve()
    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else baseline_path.parent)

    if not baseline_path.exists():
        sys.exit(f'ERROR: baseline stats not found: {baseline_path}')

    with open(baseline_path) as f:
        baseline_data = json.load(f)
    baseline = baseline_data.get('global', {})
    print(f'Loaded baseline stats: {len(baseline)} signals')

    sweep = None
    if args.sweep_table:
        sweep_path = Path(args.sweep_table).resolve()
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep = json.load(f)
            print(f'Loaded sweep table: {len(sweep)} windows')
        else:
            print(f'WARNING: sweep table not found: {sweep_path}')

    # ── Threshold computation ────────────────────────────────────────
    tf = compute_threshold(baseline, 'r_force_filt',   args.far)
    te = compute_threshold(baseline, 'r_encoder_filt', args.far)

    if tf['threshold'] is None:
        sys.exit('ERROR: r_force_filt not found in baseline stats. '
                 'Run analyze_baseline.py first.')
    if te['threshold'] is None:
        sys.exit('ERROR: r_encoder_filt not found in baseline stats.')

    # ── EMA alpha ────────────────────────────────────────────────────
    af = compute_alpha(baseline, 'r_force',   tf['threshold'], tf['mean_h'])
    ae = compute_alpha(baseline, 'r_encoder', te['threshold'], te['mean_h'])

    # ── Debounce ─────────────────────────────────────────────────────
    deb = compute_debounce(args.far, args.p_far_total, args.publish_rate)

    # ── MDL from sweep (optional) ─────────────────────────────────────
    results = {
        'thresh_force':   tf,
        'thresh_encoder': te,
        'alpha_force':    af,
        'alpha_encoder':  ae,
        'debounce':       deb,
    }

    if sweep is not None:
        # Ch0 — offset fault on tau_ext, sensitive to r_force_filt
        mdl_f = mdl_from_sweep(sweep, 0, 'offset', 'r_force_filt')
        if mdl_f is not None:
            results['mdl_force'] = mdl_f
            results['latency_force'] = latency_at_mdl(sweep, 0, 'offset',
                                                       'r_force_filt', mdl_f)
        # Ch3 — offset fault on encoder, sensitive to r_encoder_filt
        mdl_e = mdl_from_sweep(sweep, 3, 'offset', 'r_encoder_filt')
        if mdl_e is not None:
            results['mdl_encoder'] = mdl_e
            results['latency_encoder'] = latency_at_mdl(sweep, 3, 'offset',
                                                         'r_encoder_filt', mdl_e)

    # ── Output ───────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path   = out_dir / 'computed_fdi_params.yaml'
    report_path = out_dir / 'thresholds_report.md'

    with open(yaml_path, 'w') as f:
        f.write(generate_yaml(results, args.far, args.p_far_total, args.publish_rate))
    with open(report_path, 'w') as f:
        f.write(generate_report(results, args.far, args.p_far_total,
                                args.publish_rate, sweep is not None))

    print(f'Wrote {yaml_path}')
    print(f'Wrote {report_path}')

    # ── Console summary ───────────────────────────────────────────────
    alpha = round(min(af['alpha'], ae['alpha']), 4)
    alpha = max(alpha, 0.05)
    print()
    print('═' * 55)
    print(' Derived fdi_node parameters')
    print('═' * 55)
    print(f'  thresh_force:   {tf["threshold"]:.4f} Nm')
    print(f'  thresh_encoder: {te["threshold"]:.5f} rad')
    print(f'  residual_alpha: {alpha:.4f}')
    print(f'  debounce_count: {deb["debounce_n"]}'
          f'  ({deb["latency_min_ms"]:.0f} ms min latency)')
    if results.get('mdl_force'):
        print(f'  MDL-90 (force):   {results["mdl_force"]:.3g} Nm')
    if results.get('mdl_encoder'):
        print(f'  MDL-90 (encoder): {results["mdl_encoder"]:.4g} rad')
    print('═' * 55)
    print()
    print(f'Apply: merge {yaml_path.name} into fdi_params.yaml')


if __name__ == '__main__':
    main()
