#!/usr/bin/env python3
"""
Standalone closure-residual analysis — Reviewer 1 Point 6.

Runs the reduced exoskeleton dynamics and records, at every step:
  - θ, θ̇, time
  - total closure norm  ‖e‖
  - per-pair closure norms ‖eᵢ‖ (4 kinematic loop constraints)
  - raw x/y/z components of each pair's error

This lets us answer the reviewer's question:
    "Does the monotonic residual growth in Fig. 5 indicate slow accumulation
     of constraint violation over the simulation horizon?"

Answer: NO — the residual is purely geometric (function of θ, not of time).
        This script demonstrates it by correlating residual with θ rather
        than with simulation time.

Output (--output-dir):
    closure_residual.csv        — per-step time series
    params.json                 — simulation parameters

Usage:
    python3 run_closure_analysis.py
    python3 run_closure_analysis.py --duration 30 --amplitude 3.5
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Dynamics class (self-contained, shares logic with dynamics_paper.py)
# ---------------------------------------------------------------------------

class ExoClosureAnalysis:
    """Reduced exoskeleton dynamics with per-pair closure residual tracking."""

    CLOSURE_PAIR_NAMES = [
        ('frame_AC_end',   'frame_rod_end'),
        ('frame_AC_end_2', 'frame_BC_end'),
        ('frame_CE_end',   'frame_BC_end_2'),
        ('frame_CE_end_2', 'slider_t'),
    ]

    DEP_JOINT_NAMES = [
        'rev_body2linkAC', 'rev_crank2shaft', 'slider',
        'rev_slider2linkBC', 'rev_linkAC2linkCE',
        'rev_palmo2prossimale', 'rev_prossimale2mediale', 'slider2',
    ]

    def __init__(self, urdf_path: str, dt: float = 0.001,
                 fric_coul: float = 2.0, fric_visc: float = 2.0,
                 fric_eps: float = 0.005, damping_theta: float = 1.0,
                 motor_inertia: float = 0.1,
                 closure_tol: float = 1e-5, max_nfev: int = 150,
                 theta_min: float = -0.75, theta_max: float = 0.09):
        self.dt = dt
        self.fric_coul = fric_coul
        self.fric_visc = fric_visc
        self.fric_eps = fric_eps
        self.damping_theta = damping_theta
        self.motor_inertia = motor_inertia
        self.closure_tol = closure_tol
        self.max_nfev = max_nfev
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.at_limit = False

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv

        self.jid_crank = self.model.getJointId('rev_crank')
        self.idx_theta = self.jid_crank - 1

        self.closure_frame_pairs = []
        for a, b in self.CLOSURE_PAIR_NAMES:
            ida = self.model.getFrameId(a)
            idb = self.model.getFrameId(b)
            if ida >= 0 and idb >= 0:
                self.closure_frame_pairs.append((ida, idb))

        dep_ids = [self.model.getJointId(n) for n in self.DEP_JOINT_NAMES]
        self.idx_opt = [jid - 1 for jid in dep_ids]

        self.lower = np.array([-2.5, -2.5, -0.015, -2.5, -2.5, -1.5, -2.5, -0.008])
        self.upper = np.array([ 2.5,  2.5,  0.004,  2.5,  2.5,  1.2,  2.5,  0.008])

        self.theta = 0.0
        self.theta_dot = 0.0
        self.theta_ddot = 0.0
        self.q = pin.neutral(self.model).copy()
        self.dq = np.zeros(self.nv)
        self.ddq = np.zeros(self.nv)
        self.last_x = np.zeros(len(self.idx_opt))
        self.B = np.zeros(self.nv)
        self.B_prev = np.zeros(self.nv)
        self.Bdot = np.zeros(self.nv)

        # Per-pair residual storage
        n = len(self.closure_frame_pairs)
        self.closure_norm_total = 0.0
        self.closure_norm_per_pair = np.zeros(n)
        self.closure_components = np.zeros(n * 3)  # raw [x,y,z] per pair

        # Initialise closure
        q0, ok = self._solve_closure(self.theta)
        if q0 is not None:
            self.q = q0
            self.B = self._compute_B(q0)
            self._compute_per_pair_residuals(q0)

    # ------------------------------------------------------------------

    def _closure_error(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        e = []
        for ida, idb in self.closure_frame_pairs:
            e.extend(self.data.oMf[ida].translation - self.data.oMf[idb].translation)
        return np.array(e)

    def _solve_closure(self, theta):
        q = self.q.copy()
        q[self.idx_theta] = theta

        def residuals(x):
            q_loc = q.copy()
            for k, idx in enumerate(self.idx_opt):
                q_loc[idx] = x[k]
            return self._closure_error(q_loc)

        sol = least_squares(
            residuals, self.last_x.copy(),
            method='trf', bounds=(self.lower, self.upper),
            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=self.max_nfev,
        )
        if not sol.success:
            return None, False

        q_sol = q.copy()
        for k, idx in enumerate(self.idx_opt):
            q_sol[idx] = sol.x[k]

        norm = float(np.linalg.norm(self._closure_error(q_sol)))
        if norm > self.closure_tol:
            return None, False

        self.last_x = sol.x.copy()
        return q_sol, sol.nfev

    def _constraint_jacobian_cond(self, q):
        """Condition number of the constraint Jacobian A = [JA-JB for each pair]."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        A_rows = []
        for ida, idb in self.closure_frame_pairs:
            JA = pin.computeFrameJacobian(
                self.model, self.data, q, ida,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            JB = pin.computeFrameJacobian(
                self.model, self.data, q, idb,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            A_rows.append(JA - JB)
        A = np.vstack(A_rows)
        return float(np.linalg.cond(A))

    def _compute_B(self, q):
        A_rows = []
        for ida, idb in self.closure_frame_pairs:
            JA = pin.computeFrameJacobian(
                self.model, self.data, q, ida, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            JB = pin.computeFrameJacobian(
                self.model, self.data, q, idb, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            A_rows.append(JA - JB)
        A = np.vstack(A_rows)
        e_th = np.zeros(self.nv)
        e_th[self.idx_theta] = 1.0
        rhs = -A @ e_th
        mask = np.ones(self.nv, dtype=bool)
        mask[self.idx_theta] = False
        x_red, *_ = np.linalg.lstsq(A[:, mask], rhs, rcond=None)
        x = np.zeros(self.nv)
        x[mask] = x_red
        return e_th + x

    def _compute_per_pair_residuals(self, q):
        """Fill self.closure_norm_per_pair and self.closure_components from q."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        total_sq = 0.0
        for i, (ida, idb) in enumerate(self.closure_frame_pairs):
            e = self.data.oMf[ida].translation - self.data.oMf[idb].translation
            n = float(np.linalg.norm(e))
            self.closure_norm_per_pair[i] = n
            self.closure_components[3 * i: 3 * i + 3] = e
            total_sq += n * n
        self.closure_norm_total = math.sqrt(total_sq)

    def step(self, tau_m: float) -> dict:
        q_new, ok = self._solve_closure(self.theta)
        if q_new is None:
            return None

        self.q = q_new
        self.B_prev = self.B.copy()
        self.B = self._compute_B(self.q)
        self.Bdot = (self.B - self.B_prev) / self.dt

        self._compute_per_pair_residuals(self.q)

        self.dq = self.B * self.theta_dot
        self.ddq = self.B * self.theta_ddot + self.Bdot * self.theta_dot

        M = pin.crba(self.model, self.data, self.q)
        h = pin.nonLinearEffects(self.model, self.data, self.q, self.dq)

        denom = float(self.B.T @ (M @ self.B)) + self.motor_inertia
        proj = float(self.B.T @ (M @ (self.Bdot * self.theta_dot) + h))

        tau_fric = (self.fric_visc * self.theta_dot
                    + self.fric_coul * np.tanh(self.theta_dot / max(self.fric_eps, 1e-9)))
        tau_damp = self.damping_theta * self.theta_dot

        num = tau_m - proj - float(tau_fric) - tau_damp
        self.theta_ddot = float(np.clip(num / denom, -400.0, 400.0))

        self.theta_dot = float(np.clip(
            self.theta_dot + self.theta_ddot * self.dt, -10.0, 10.0))
        self.theta = float(self.theta + self.theta_dot * self.dt)

        # Enforce physical theta limits (elastic rebound)
        if self.theta <= self.theta_min:
            self.theta = self.theta_min
            self.theta_dot = max(0.0, self.theta_dot)
            self.at_limit = True
        elif self.theta >= self.theta_max:
            self.theta = self.theta_max
            self.theta_dot = min(0.0, self.theta_dot)
            self.at_limit = True
        else:
            self.at_limit = False

        out = {
            'theta':          float(self.theta),
            'theta_dot':      float(self.theta_dot),
            'closure_total':  self.closure_norm_total,
        }
        for i in range(len(self.closure_frame_pairs)):
            out[f'pair{i}_norm'] = float(self.closure_norm_per_pair[i])
            out[f'pair{i}_ex']   = float(self.closure_components[3 * i])
            out[f'pair{i}_ey']   = float(self.closure_components[3 * i + 1])
            out[f'pair{i}_ez']   = float(self.closure_components[3 * i + 2])
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Record per-pair closure residuals vs theta and time'
    )
    parser.add_argument('--urdf', type=str, default=None)
    parser.add_argument('--mode', type=str, default='scan',
                        choices=['scan', 'dynamics'],
                        help='"scan": quasi-static theta sweep (default); '
                             '"dynamics": half-sine driven simulation')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='[dynamics] Simulation duration [s] (default: 30)')
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--warmup', type=float, default=1.0)
    parser.add_argument('--amplitude', type=float, default=3.5,
                        help='[dynamics] Half-sine torque amplitude [Nm] (default: 3.5)')
    parser.add_argument('--period', type=float, default=10.0,
                        help='[dynamics] Torque period [s] (default: 10.0)')
    parser.add_argument('--theta-min', type=float, default=-0.75,
                        help='Physical lower limit of theta [rad] (default: -0.75)')
    parser.add_argument('--theta-max', type=float, default=0.09,
                        help='Physical upper limit of theta [rad] (default: 0.09)')
    parser.add_argument('--scan-steps', type=int, default=1000,
                        help='[scan] Number of steps per sweep direction (default: 1000)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.urdf is None:
        args.urdf = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'urdf', 'assembly_with_hand.urdf'))
    if args.output_dir is None:
        args.output_dir = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'results', 'closure_residual'))

    os.makedirs(args.output_dir, exist_ok=True)

    n_pairs = 4  # known from the mechanism
    pair_labels = [
        'AC_end↔rod_end',
        'AC_end_2↔BC_end',
        'CE_end↔BC_end_2',
        'CE_end_2↔slider_t',
    ]

    print(f'URDF:       {args.urdf}')
    print(f'Mode:       {args.mode}')
    print(f'Theta:      [{args.theta_min}, {args.theta_max}] rad')
    print(f'Output:     {args.output_dir}')
    print()

    sim = ExoClosureAnalysis(args.urdf, dt=args.dt,
                             theta_min=args.theta_min, theta_max=args.theta_max)

    fields = ['step', 'time', 'theta', 'theta_dot', 'tau_m', 'direction',
              'closure_total', 'nfev', 'cond_A']
    for i in range(n_pairs):
        fields.append(f'pair{i}_norm')
    for i in range(n_pairs):
        fields += [f'pair{i}_ex', f'pair{i}_ey', f'pair{i}_ez']

    csv_path = os.path.join(args.output_dir, 'closure_residual.csv')
    t_start = time.time()

    if args.mode == 'scan':
        # -----------------------------------------------------------
        # Quasi-static sweep: θ from theta_max → theta_min → theta_max
        # No dynamics — just solve closure at each θ position.
        # Forward and return pass overlay perfectly iff residual is geometric.
        # -----------------------------------------------------------
        print(f'Scan steps: {args.scan_steps} per direction (total {2*args.scan_steps})')
        theta_fwd = np.linspace(args.theta_max, args.theta_min, args.scan_steps)
        theta_rev = np.linspace(args.theta_min, args.theta_max, args.scan_steps)
        theta_all = np.concatenate([theta_fwd, theta_rev])
        dirs = ['fwd'] * args.scan_steps + ['rev'] * args.scan_steps

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for idx, (th, dr) in enumerate(zip(theta_all, dirs)):
                q_sol, nfev = sim._solve_closure(th)
                if q_sol is None:
                    print(f'  [idx {idx}] closure failed at theta={th:.4f}, skipping.')
                    continue
                sim.q = q_sol
                sim._compute_per_pair_residuals(q_sol)
                cond_A = sim._constraint_jacobian_cond(q_sol)

                row = {
                    'step':          idx,
                    'time':          float(idx) * args.dt,
                    'theta':         float(th),
                    'theta_dot':     0.0,
                    'tau_m':         0.0,
                    'direction':     dr,
                    'closure_total': sim.closure_norm_total,
                    'nfev':          nfev,
                    'cond_A':        cond_A,
                }
                for i in range(len(sim.closure_frame_pairs)):
                    row[f'pair{i}_norm'] = float(sim.closure_norm_per_pair[i])
                    row[f'pair{i}_ex']   = float(sim.closure_components[3 * i])
                    row[f'pair{i}_ey']   = float(sim.closure_components[3 * i + 1])
                    row[f'pair{i}_ez']   = float(sim.closure_components[3 * i + 2])
                writer.writerow(row)

                if (idx + 1) % 200 == 0:
                    print(f'  idx {idx+1:4d}/{len(theta_all)}  '
                          f'theta={th:+.4f} [{dr}]  '
                          f'closure_total={sim.closure_norm_total:.3e}')

        params = {
            'urdf':        os.path.abspath(args.urdf),
            'mode':        'scan',
            'scan_steps':  args.scan_steps,
            'theta_min':   args.theta_min,
            'theta_max':   args.theta_max,
            'pair_labels': pair_labels,
            'amplitude':   None,
            'period':      None,
        }

    else:
        # -----------------------------------------------------------
        # Dynamics mode: half-sine negative torque
        # -----------------------------------------------------------
        def tau_fn(t):
            return -abs(args.amplitude) * abs(math.sin(math.pi * t / args.period))

        print(f'Duration:   {args.duration} s   dt={args.dt*1e3:.1f} ms')
        print(f'Torque:     half-sine neg, A={args.amplitude} Nm, T={args.period} s')

        total_steps = int(args.duration / args.dt)
        warmup_steps = int(args.warmup / args.dt)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()

            for i in range(total_steps):
                t = i * args.dt
                tau_m = tau_fn(t)
                result = sim.step(tau_m)
                if result is None:
                    print(f'  [step {i}] closure failed, stopping.')
                    break

                if i < warmup_steps:
                    continue

                row = {'step': i, 'time': t, 'tau_m': tau_m, 'direction': 'fwd'}
                row.update(result)
                writer.writerow(row)

                if (i + 1) % 5000 == 0:
                    print(f'  step {i+1:6d}/{total_steps}  '
                          f'theta={result["theta"]:+.4f} rad  '
                          f'closure_total={result["closure_total"]:.3e}')

        params = {
            'urdf':          os.path.abspath(args.urdf),
            'mode':          'dynamics',
            'duration':      args.duration,
            'dt':            args.dt,
            'warmup':        args.warmup,
            'amplitude':     args.amplitude,
            'period':        args.period,
            'profile':       'half_sine_neg',
            'pair_labels':   pair_labels,
            'theta_min':     args.theta_min,
            'theta_max':     args.theta_max,
            'fric_coul':     2.0,
            'fric_visc':     2.0,
            'fric_eps':      0.005,
            'damping_theta': 1.0,
        }

    t_elapsed = time.time() - t_start
    print(f'\nWall clock: {t_elapsed:.1f} s')
    print(f'Saved:      {csv_path}')

    with open(os.path.join(args.output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    print('Done.')


if __name__ == '__main__':
    main()
