# Closure residual analysis — Reviewer 1 Point 6

**Date:** 2026-05-09  
**Addresses:** Reviewer 1 – Point 6 of `modifiche_paper.md`  
*(Il residuo di chiusura cresce monotonicamente da 0 a circa 4×10⁻⁶ in Fig.5 — ciò indica un accumulo lento di violazione dei vincoli?)*

---

## 1. Reviewer question

> In Fig. 5 il residuo cresce monotonicamente da 0 a circa 4×10⁻⁶. Benché sotto la tolleranza, spiegare se ciò indica un accumulo lento di violazione dei vincoli sull'orizzonte di simulazione.

**Short answer:** No. The monotonic growth seen in Fig. 5 is **purely geometric** — the residual at the solver optimum is a deterministic function of θ, not an accumulation of constraint violation over time. This is demonstrated below.

---

## 2. Physical setup

The exoskeleton mechanism has 4 kinematic loop constraints, each enforced by a frame-pair coincidence condition:

| Index | Pair |
|-------|------|
| 0 | `frame_AC_end` ↔ `frame_rod_end` |
| 1 | `frame_AC_end_2` ↔ `frame_BC_end` |
| 2 | `frame_CE_end` ↔ `frame_BC_end_2` |
| 3 | `frame_CE_end_2` ↔ `slider_t` |

At each integration step the solver minimises `‖e(q,θ)‖` over the 8 dependent joint coordinates with a trust-region least-squares algorithm (`scipy.optimize.least_squares`, TRF method). The solver tolerance is `ftol = xtol = gtol = 1e-8`, and a step is accepted only if `‖e‖ < 1e-5 m`.

---

## 3. Experiment design

To isolate the question of *geometric vs temporal* residual growth, a **quasi-static sweep** was performed instead of a dynamic simulation:

- θ was varied **quasi-statically** from θ_max = 0.09 rad to θ_min = −0.75 rad (**forward sweep**) and then back from θ_min to θ_max (**return sweep**), using 1000 uniform steps per direction.
- At each θ position the closure solver was called directly (no dynamics, no warm-start from previous velocity).
- If the residual were accumulated over time, the return sweep would show a systematically higher residual than the forward sweep. If it is geometric, the two sweeps overlay exactly.

| Parameter | Value |
|-----------|-------|
| Mode | Quasi-static scan |
| θ range | [−0.75, 0.09] rad (physical limits) |
| Steps per direction | 1000 |
| Total points | 2000 |
| URDF | `urdf/assembly_with_hand.urdf` (9 DOF, 4 closure constraints) |
| Solver | TRF least-squares, tol = 1e-8, max_nfev = 150 |
| Solver acceptance threshold | ‖e‖ < 1e-5 m |

---

## 4. Results

### 4.1 Residual magnitudes

| Constraint pair | Mean ‖eᵢ‖ [m] | Max ‖eᵢ‖ [m] | Max / tolerance |
|-----------------|---------------|--------------|-----------------|
| **TOTAL ‖e‖** | **2.409 × 10⁻⁶** | **5.045 × 10⁻⁶** | **0.50×** |
| AC_end ↔ rod_end | 2.50 × 10⁻⁸ | 6.30 × 10⁻⁸ | 0.01× |
| AC_end_2 ↔ BC_end | 1.70 × 10⁻⁶ | 3.56 × 10⁻⁶ | 0.36× |
| CE_end ↔ BC_end_2 | 1.70 × 10⁻⁶ | 3.57 × 10⁻⁶ | 0.36× |
| CE_end_2 ↔ slider_t | 9.20 × 10⁻⁸ | 1.61 × 10⁻⁷ | 0.02× |

- The total residual peaks at **5.045 × 10⁻⁶ m** at θ = −0.75 rad, i.e. **50% of the acceptance threshold**.
- Pairs 1 and 2 (the two crank-link connections in the central four-bar loop) carry almost all the residual; pairs 0 and 3 are negligible.

### 4.2 Forward / return overlay (geometric test)

| Metric | Value |
|--------|-------|
| Max |fwd − rev| | **2.39 × 10⁻⁸ m** |
| Relative to peak residual | **0.47 %** |
| Conclusion | Forward and return sweeps **identical to < 1%** |

The forward and return passes differ by at most **2.4 × 10⁻⁸ m**, which is two orders of magnitude below the solver tolerance. This confirms that the residual is a **pure function of θ** — it has no memory of the path taken to reach that θ value.

### 4.3 Correlation analysis

| Metric | Value |
|--------|-------|
| \|corr(‖e‖, θ)\| | **0.9922** |

The correlation of total residual with θ is 0.992, confirming a near-perfect monotone mapping e(θ). The slight deviation from 1.0 is due to the non-linearity of the mapping (the relationship is smooth but not linear).

---

## 5. Why Fig. 5 shows monotonic growth

During the simulated trial in Fig. 5, the motor drives the exoskeleton through a single opening stroke: θ decreases monotonically from ~0 toward ~−0.75 rad. Because e(θ) is a monotone increasing function of |θ|, the residual also appears to grow monotonically with time.

This is **not constraint violation accumulation** — it is a projection effect. If the motor reversed direction, the residual would follow the same curve back toward zero, which is exactly what the return sweep demonstrates (0.47% overlay error).

---

## 6. Root-cause analysis: URDF geometric inconsistency

To identify *why* the residual grows with |θ|, three quantities were analysed across the sweep (`fig_cause_analysis.pdf`):

### 6.1 Error component decomposition

Decomposing the closure error into x/y/z components reveals that the error is **almost entirely in the z-direction** (perpendicular to the plane of the mechanism):

| Pair | eₓ range [µm] | e_y range [µm] | e_z range [µm] |
|------|--------------|---------------|---------------|
| 1 — AC_end_2 ↔ BC_end   | [−0.04, +0.23] | [−0.41, +1.30] | **[−3.35, +0.33]** |
| 2 — CE_end ↔ BC_end_2   | [−0.04, +0.24] | [−1.32, +0.40] | **[−0.36, +3.35]** |

The z-errors of pairs 1 and 2 are **equal in magnitude and opposite in sign** across the entire θ range. This anti-symmetry is a direct signature of a small **out-of-plane offset of the BC link** in the URDF: BC_end and BC_end_2 are both connected to the BC link body, so a z-misalignment of that body produces equal and opposite errors on whichever two chains close through it.

### 6.2 Solver convergence (nfev)

The number of solver function evaluations is **≈ 2 throughout the entire sweep** (mean 2.05, max 7 at two isolated configurations). The solver always finds the optimum in near-minimum iterations — the residual is not caused by premature convergence.

### 6.3 Constraint Jacobian conditioning

The smallest non-zero singular value of the dependent-DOF constraint Jacobian A_dep (12 × 8) decreases from ~0.006 at θ = 0.09 to ~0.003 at θ = −0.75 — a factor of **2×**. The Jacobian is always rank-7 out of 8, reflecting one kinematically unconstrained mode (expected for this mechanism topology). The modest conditioning change is a secondary effect; it does not drive the residual growth.

### 6.4 Conclusion: URDF parameter error in z

The root cause is a **small geometric inconsistency in the URDF specification of the BC link**, most likely a z-axis offset introduced during CAD export or manual joint placement. At θ ≈ 0 (the CAD assembly reference configuration) the error is near-zero. As θ moves away from 0, the accumulated angular displacement translates this z-offset into a growing translational gap at the closure frames, producing the linear-in-θ residual growth seen in Fig. 5. The solver correctly finds the least-squares optimum at each configuration; there is no constraint violation in the integration sense.

---

## 7. Generated files

| File | Description |
|------|-------------|
| `closure_residual.csv` | Per-step data: θ, direction, closure_total, nfev, cond_A, per-pair norms and components |
| `params.json` | Scan parameters |
| `fig_geometric_proof.pdf/png` | **Key figure**: forward and return sweeps overlaid vs θ |
| `fig_cause_analysis.pdf/png` | Error components + conditioning + nfev vs θ (root-cause figure) |
| `fig_closure_vs_theta.pdf/png` | Total + per-pair ‖eᵢ‖ vs θ |
| `fig_closure_vs_time.pdf/png` | Total and per-pair residual vs scan index |
| `fig_closure_components.pdf/png` | Raw x/y/z error components per pair |
| `closure_summary.txt` | Numerical summary with overlay test |

---

## 8. Paper paragraph (Reviewer 1 – Point 6)

> The closure residual shown in Fig. 5 grows monotonically from zero to approximately 5×10⁻⁶ m because the residual at the solver optimum is a smooth, deterministic function of the crank angle θ — not an accumulation of constraint violation over the simulation horizon. To verify this, we performed a quasi-static scan: θ was swept uniformly from θ_max = 0.09 rad to θ_min = −0.75 rad (the physical range of the mechanism) and back, calling the kinematic closure solver independently at each position. The forward and return passes overlay to within **2.4 × 10⁻⁸ m** (0.47% of the peak residual), confirming that the residual carries no path history. Decomposing the error into Cartesian components reveals that it lies almost entirely in the z-direction (perpendicular to the mechanism plane), with pairs 1 and 2 — both connected through the BC link — showing equal and opposite z-errors. This anti-symmetry identifies the root cause as a small out-of-plane offset in the URDF specification of the BC link, most likely introduced during CAD export. The solver converges in a mean of 2.05 function evaluations across the full θ range, confirming that the residual reflects the geometry of the URDF rather than premature solver termination. The maximum residual is **5.05 × 10⁻⁶ m** (50% of the solver tolerance of 10⁻⁵ m); no constraint is violated in the integration sense.
