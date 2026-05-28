#!/usr/bin/env python3
"""
diagnose_wall_contact.py — quantify how often theta hit the virtual walls
==========================================================================

Reads a baseline (or fault sweep) bag and computes:

  - fraction of time theta was inside [theta_min + wall_buffer,
                                       theta_max - wall_buffer]
    → "free zone": no wall force, Luenberger model is driven only
      by the commanded torque
  - fraction of time theta was inside the buffer zone
    → "buffer zone": tau_wall is active, adding an unmodelled
      disturbance that shifts the Luenberger state residual
  - fraction of time theta was clamped at the limits
    → "saturation": worst case for the Luenberger

It also reports the same statistics for theta_v (the admittance virtual
trajectory), and correlates wall-contact time with the magnitude of
the state residual r_state.

Usage
-----
    python3 diagnose_wall_contact.py --bag /path/to/bag_dir

Output is plain text on stdout.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import rclpy.serialization as ser
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f'ERROR: ROS 2 Python bindings not available ({e})', file=sys.stderr)
    print('       Source the workspace before running this script.', file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Configuration: physical limits and admittance wall parameters.
# These MUST match the actual values used during the recording.
# ─────────────────────────────────────────────────────────────────
THETA_MIN     = -0.75   # [rad] lower mechanical limit
THETA_MAX     =  0.09   # [rad] upper mechanical limit
WALL_BUFFER   =  0.05   # [rad] buffer width before the wall starts
JOINT_NAME    = 'rev_crank'


def _detect_storage_id(bag_path: str) -> str:
    p = Path(bag_path)
    if any(p.glob('*.mcap')):
        return 'mcap'
    if any(p.glob('*.db3')):
        return 'sqlite3'
    return 'mcap'


def read_signals(bag_path: str) -> dict:
    """
    Returns a dict with keys:
      'theta'              from /joint_states
      'theta_v'            from /trajectory_ref[0]
      'theta_dot_v'        from /trajectory_ref[1]
      'r_state'            from /observer/state_residual

    Each value is a np.ndarray of samples (timestamps are not kept,
    we only need statistics).
    """
    storage = StorageOptions(uri=bag_path, storage_id=_detect_storage_id(bag_path))
    converter = ConverterOptions(input_serialization_format='cdr',
                                 output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage, converter)
    msg_classes = {t.name: get_message(t.type)
                   for t in reader.get_all_topics_and_types()}

    out = defaultdict(list)

    while reader.has_next():
        topic, raw, t = reader.read_next()
        cls = msg_classes.get(topic)
        if cls is None:
            continue

        if topic == '/joint_states':
            msg = ser.deserialize_message(raw, cls)
            try:
                idx = list(msg.name).index(JOINT_NAME)
                out['theta'].append(float(msg.position[idx]))
            except ValueError:
                pass
        elif topic == '/trajectory_ref':
            msg = ser.deserialize_message(raw, cls)
            if len(msg.data) >= 2:
                out['theta_v'].append(float(msg.data[0]))
                out['theta_dot_v'].append(float(msg.data[1]))
        elif topic == '/observer/state_residual':
            msg = ser.deserialize_message(raw, cls)
            out['r_state'].append(float(msg.data))

    return {k: np.array(v) for k, v in out.items()}


def fraction_in_zone(values: np.ndarray, lo: float, hi: float) -> float:
    if len(values) == 0:
        return 0.0
    mask = (values >= lo) & (values <= hi)
    return float(mask.sum()) / len(values)


def report_signal(name: str, vals: np.ndarray):
    if len(vals) == 0:
        print(f'\n[{name}]  (empty — topic not found in bag)')
        return

    free_lo = THETA_MIN + WALL_BUFFER
    free_hi = THETA_MAX - WALL_BUFFER

    f_free   = fraction_in_zone(vals, free_lo, free_hi)
    f_buf_lo = fraction_in_zone(vals, THETA_MIN, free_lo)
    f_buf_hi = fraction_in_zone(vals, free_hi, THETA_MAX)
    f_low    = fraction_in_zone(vals, -np.inf, THETA_MIN)
    f_high   = fraction_in_zone(vals, THETA_MAX,  np.inf)

    print(f'\n[{name}]')
    print(f'  N samples       : {len(vals)}')
    print(f'  range           : [{vals.min():+.4f}, {vals.max():+.4f}] rad')
    print(f'  mean ± std      : {vals.mean():+.4f} ± {vals.std():.4f} rad')
    print()
    print(f'  Wall regions (limits = [{THETA_MIN}, {THETA_MAX}], buffer={WALL_BUFFER}):')
    print(f'    free zone     [{free_lo:+.3f}, {free_hi:+.3f}] : {100*f_free:6.2f} %')
    print(f'    lower buffer  [{THETA_MIN:+.3f}, {free_lo:+.3f}] : {100*f_buf_lo:6.2f} %')
    print(f'    upper buffer  [{free_hi:+.3f}, {THETA_MAX:+.3f}] : {100*f_buf_hi:6.2f} %')
    print(f'    below lower   < {THETA_MIN:+.3f}                   : {100*f_low:6.2f} %')
    print(f'    above upper   > {THETA_MAX:+.3f}                   : {100*f_high:6.2f} %')

    in_buffer = f_buf_lo + f_buf_hi
    in_violation = f_low + f_high
    print(f'  TOTAL in buffer zone    : {100*in_buffer:6.2f} %')
    print(f'  TOTAL beyond limits     : {100*in_violation:6.2f} %  (should be 0 for theta_v)')


def correlation_residual_with_wall(theta: np.ndarray, residual: np.ndarray,
                                    name: str):
    """
    Compare residual statistics on samples that are FAR from the wall
    versus samples close to the wall. If the wall is biasing the Luenberger
    we'll see a clear difference in the std (or mean) of the residual.
    """
    if len(theta) == 0 or len(residual) == 0:
        return
    # Trim to common length (the topics have slightly different rates)
    n = min(len(theta), len(residual))
    th = theta[:n]
    rs = residual[:n]

    free_lo = THETA_MIN + WALL_BUFFER
    free_hi = THETA_MAX - WALL_BUFFER

    free_mask = (th >= free_lo) & (th <= free_hi)
    near_wall_mask = ~free_mask

    if free_mask.sum() < 100 or near_wall_mask.sum() < 100:
        print(f'\n[{name}: cross-correlation]  not enough samples in '
              f'one of the zones (free={free_mask.sum()}, '
              f'near={near_wall_mask.sum()})')
        return

    print(f'\n[{name}: residual statistics by zone]')
    print(f'  Free zone     ({free_mask.sum():>7d} samples)  '
          f'mean={rs[free_mask].mean():+.5f}  '
          f'std={rs[free_mask].std():.5f}  '
          f'p99.9(|r|)={np.percentile(np.abs(rs[free_mask] - rs[free_mask].mean()), 99.9):.5f}')
    print(f'  Near wall     ({near_wall_mask.sum():>7d} samples)  '
          f'mean={rs[near_wall_mask].mean():+.5f}  '
          f'std={rs[near_wall_mask].std():.5f}  '
          f'p99.9(|r|)={np.percentile(np.abs(rs[near_wall_mask] - rs[near_wall_mask].mean()), 99.9):.5f}')


def main():
    # NB: 'global' must come BEFORE any use of these names in this scope,
    # otherwise Python flags it as 'used prior to global declaration'.
    global THETA_MIN, THETA_MAX, WALL_BUFFER

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', required=True, help='rosbag2 directory')
    ap.add_argument('--theta-min', type=float, default=THETA_MIN)
    ap.add_argument('--theta-max', type=float, default=THETA_MAX)
    ap.add_argument('--wall-buffer', type=float, default=WALL_BUFFER)
    args = ap.parse_args()

    THETA_MIN = args.theta_min
    THETA_MAX = args.theta_max
    WALL_BUFFER = args.wall_buffer

    bag = Path(args.bag).resolve()
    if not bag.exists():
        sys.exit(f'Bag not found: {bag}')

    print(f'Reading bag: {bag}')
    print(f'Limits:  [{THETA_MIN}, {THETA_MAX}] rad')
    print(f'Buffer:  {WALL_BUFFER} rad')
    print(f'(this may take a minute on a large bag — be patient)')

    data = read_signals(str(bag))

    print(f'\nExtracted signals: {sorted(data.keys())}')

    # Report theta and theta_v separately. theta_v is what the admittance
    # virtual trajectory reaches; theta is what the plant actually does.
    # If theta_v is in the buffer it means the wall force is being applied.
    report_signal('theta (plant)', data.get('theta', np.array([])))
    report_signal('theta_v (admittance virtual)', data.get('theta_v', np.array([])))

    # Correlate residual magnitude with wall proximity.
    correlation_residual_with_wall(
        data.get('theta_v', np.array([])),
        data.get('r_state', np.array([])),
        'r_state',
    )

    # ── Summary verdict ──
    theta_v = data.get('theta_v', np.array([]))
    if len(theta_v) > 0:
        free_lo = THETA_MIN + WALL_BUFFER
        free_hi = THETA_MAX - WALL_BUFFER
        free_frac = fraction_in_zone(theta_v, free_lo, free_hi)
        print(f'\n{"="*60}')
        print(f'VERDICT')
        print(f'{"="*60}')
        if free_frac > 0.95:
            print(f'  theta_v is in the free zone {100*free_frac:.1f}% of the time.')
            print(f'  → Wall contact is sporadic. Bag is likely USABLE.')
            print(f'    Proceed with Luenberger threshold calibration using')
            print(f'    the full baseline data.')
        elif free_frac > 0.80:
            print(f'  theta_v is in the free zone {100*free_frac:.1f}% of the time.')
            print(f'  → Some wall contact. Bag is RECOVERABLE: we should')
            print(f'    update analyze_baseline.py to FILTER OUT samples')
            print(f'    where theta_v is in the buffer zone, then')
            print(f'    recompute statistics.')
        else:
            print(f'  theta_v is in the free zone only {100*free_frac:.1f}% of the time.')
            print(f'  → SIGNIFICANT wall contact. Recommend re-recording')
            print(f'    the baseline with smaller wrench amplitudes')
            print(f'    (reduce f_min in healthy_regimes.yaml so that')
            print(f'    theta_v does not push against the walls).')


if __name__ == '__main__':
    main()