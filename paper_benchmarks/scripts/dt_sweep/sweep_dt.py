#!/usr/bin/env python3
"""
Generate benchmark data at multiple dt values to study the sensitivity of the
reduced dynamics to the Bdot finite-difference approximation.

Usage:
    python3 sweep_dt.py [--dt-list "0.00005,0.0001,0.0005,0.001,0.002,0.005"]
                        [--duration 20] [--amplitude 0.5] [--period 10]
                        [--output-dir ../../results/dt_sweep]

Output: one CSV per dt in output-dir, plus a params.json
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


class ExoDynamicsSweep:
    """
    Standalone reduced dynamics (no ROS) — modified for dt-sweep analysis.

    Compared to ExoDynamicsBenchmark in computation_time_step_standalone.py,
    this version:
      - uses a TIME-BASED sine torque profile (same physical signal for every dt)
      - stores B and Bdot in the step output
      - saves all quantities (theta, M_eff, proj, B, Bdot, ...) to CSV
    """

    def __init__(self, urdf_path: str, dt: float = 0.001, params: Optional[Dict] = None):
        self.dt = dt

        self.p = {
            'motor_inertia': 0.1,
            'fric_visc': 2.0,
            'fric_coul': 2.0,
            'fric_eps': 0.005,
            'damping_theta': 1.0,
            'max_theta_dot': 10.0,
            'max_theta_ddot': 400.0,
            'closure_tol': 1e-5,
            'max_nfev': 150,
            'denom_min': 1e-7,
            'theta_min': -2.5,
            'theta_max': 2.5,
            'theta_init': 0.0,
            'limit_hold': True,
            'limit_release_tau': 0.02,
            'limit_backoff_step': 0.003,
            'limit_backoff_tries': 6,
            'limit_use_theta_bounds': True,
            'wrench_enable': False,
            'wrench_frame': 'frame_CE_end_2',
            'wrench_scale': 1.0,
            'passive_enable': True,
            'K0_MCF': 0.03,
            'alpha_MCF': 4.0,
            'B_MCF': 0.01,
            'rest_MCF': -0.3,
            'K0_IFP': 0.03,
            'alpha_IFP': 4.0,
            'B_IFP': 0.01,
            'rest_IFP': -0.5,
            'passive_K_MAX': 5.0,
            'passive_exp_clip': 12.0,
            'medial_soft_enable': True,
            'medial_soft_lower': -2.2,
            'medial_soft_upper': 0.0,
            'medial_soft_weight': 1.0,
        }
        if params:
            self.p.update(params)

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.jid_crank = self.model.getJointId('rev_crank')
        self.idx_theta = self.jid_crank - 1

        self.jid_MCF = self.model.getJointId('rev_palmo2prossimale')
        self.jid_IFP = self.model.getJointId('rev_prossimale2mediale')

        self.closure_frame_name_pairs = [
            ('frame_AC_end', 'frame_rod_end'),
            ('frame_AC_end_2', 'frame_BC_end'),
            ('frame_CE_end', 'frame_BC_end_2'),
            ('frame_CE_end_2', 'slider_t'),
        ]
        self.closure_frame_pairs: List[Tuple[int, int]] = []
        for a, b in self.closure_frame_name_pairs:
            ida = self.model.getFrameId(a)
            idb = self.model.getFrameId(b)
            if ida >= 0 and idb >= 0:
                self.closure_frame_pairs.append((ida, idb))

        dep_joint_names = [
            'rev_body2linkAC', 'rev_crank2shaft', 'slider',
            'rev_slider2linkBC', 'rev_linkAC2linkCE',
            'rev_palmo2prossimale', 'rev_prossimale2mediale', 'slider2',
        ]
        dep_ids = [self.model.getJointId(n) for n in dep_joint_names]
        self.idx_opt = [jid - 1 for jid in dep_ids]

        self.lower = np.array([-2.5, -2.5, -0.015, -2.5, -2.5, -1.5, -2.5, -0.008], dtype=float)
        self.upper = np.array([2.5, 2.5, 0.004, 2.5, 2.5, 1.2, 2.5, 0.008], dtype=float)

        self.fid_ext = -1
        if self.p['wrench_enable']:
            self.fid_ext = self.model.getFrameId(self.p['wrench_frame'])
            if self.fid_ext < 0:
                self.p['wrench_enable'] = False

        self.theta = float(self.p['theta_init'])
        self.theta_dot = 0.0
        self.theta_ddot = 0.0

        self.q = pin.neutral(self.model).copy()
        self.dq = np.zeros(self.nv)
        self.ddq = np.zeros(self.nv)

        self.last_x = np.zeros(len(self.idx_opt), dtype=float)
        self.B = np.zeros(self.nv)
        self.B_prev = np.zeros(self.nv)
        self.Bdot = np.zeros(self.nv)

        self.tau_m = 0.0
        self.wrench_world = np.zeros(6)
        self.tau_ext = np.zeros(self.nv)
        self.tau_ext_theta = 0.0
        self.tau_pass = np.zeros(self.nv)
        self.tau_pass_theta = 0.0
        self.tau_full = np.zeros(self.nv)

        self.denom_last = 0.0
        self.proj_last = 0.0
        self.gproj_last = 0.0
        self.tau_fric_last = 0.0
        self.tau_damp_last = 0.0
        self.num_last = 0.0

        self.at_limit = False
        self.limit_dir = 0.0
        self.have_valid = False
        self.theta_valid = float(self.theta)
        self.q_valid = self.q.copy()
        self.dq_valid = self.dq.copy()
        self.B_valid = self.B.copy()
        self.last_x_valid = self.last_x.copy()

        q0, info0 = self.solve_closure(self.theta, update_warmstart=True)
        if q0 is not None:
            self.q = q0
            self.B = self.compute_B(self.q)
            self._store_valid_state()
        else:
            self.at_limit = True
            self.theta_dot = 0.0
            self.theta_ddot = 0.0

    def _store_valid_state(self) -> None:
        self.have_valid = True
        self.theta_valid = float(self.theta)
        self.q_valid = self.q.copy()
        self.dq_valid = self.dq.copy()
        self.B_valid = self.B.copy()
        self.last_x_valid = self.last_x.copy()

    def _clamp_theta_bounds(self, theta: float) -> float:
        if not self.p['limit_use_theta_bounds']:
            return float(theta)
        return float(np.clip(theta, self.p['theta_min'], self.p['theta_max']))

    def closure_error(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        e: List[float] = []
        for ida, idb in self.closure_frame_pairs:
            e.extend(self.data.oMf[ida].translation - self.data.oMf[idb].translation)
        return np.array(e, dtype=float)

    def solve_closure(
        self, theta: float, update_warmstart: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        info: Dict[str, float] = {
            'success': False, 'nfev': 0, 'closure_norm': np.inf, 'hit_bounds': False,
        }
        q = self.q.copy()
        q[self.idx_theta] = theta

        if len(self.closure_frame_pairs) == 0:
            cnorm = float(np.linalg.norm(self.closure_error(q)))
            info['success'] = True
            info['nfev'] = 0
            info['closure_norm'] = cnorm
            return q, info

        medial_soft = self.p['medial_soft_enable']
        medial_lower = self.p['medial_soft_lower']
        medial_upper = self.p['medial_soft_upper']
        medial_w = self.p['medial_soft_weight']
        idx_medial_in_x = 6

        def residuals(x: np.ndarray) -> np.ndarray:
            q_loc = q.copy()
            for k, idx in enumerate(self.idx_opt):
                q_loc[idx] = x[k]
            e = self.closure_error(q_loc)
            if not medial_soft:
                return e
            medial = float(x[idx_medial_in_x])
            penalty = 0.0
            if medial < medial_lower:
                penalty += (medial - medial_lower) ** 2
            if medial > medial_upper:
                penalty += (medial - medial_upper) ** 2
            return np.hstack([e, math.sqrt(medial_w * penalty)])

        sol = least_squares(
            residuals, self.last_x.copy(), method='trf',
            bounds=(self.lower, self.upper),
            xtol=1e-8, ftol=1e-8, gtol=1e-8,
            max_nfev=int(self.p['max_nfev']),
        )

        info['success'] = bool(sol.success)
        info['nfev'] = int(sol.nfev)
        epsb = 1e-6
        info['hit_bounds'] = any(
            abs(sol.x[k] - self.lower[k]) < epsb or abs(sol.x[k] - self.upper[k]) < epsb
            for k in range(len(sol.x))
        )
        if not sol.success:
            return None, info

        q_sol = q.copy()
        for k, idx in enumerate(self.idx_opt):
            q_sol[idx] = sol.x[k]

        cnorm = float(np.linalg.norm(self.closure_error(q_sol)))
        info['closure_norm'] = cnorm

        if cnorm > self.p['closure_tol']:
            info['success'] = False
            return None, info

        if update_warmstart:
            self.last_x = sol.x.copy()
        return q_sol, info

    def compute_B(self, q: np.ndarray) -> np.ndarray:
        if len(self.closure_frame_pairs) == 0:
            B = np.zeros(self.nv)
            B[self.idx_theta] = 1.0
            return B

        A_rows: List[np.ndarray] = []
        for ida, idb in self.closure_frame_pairs:
            JA = pin.computeFrameJacobian(
                self.model, self.data, q, ida, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3, :]
            JB = pin.computeFrameJacobian(
                self.model, self.data, q, idb, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3, :]
            A_rows.append(JA - JB)

        A = np.vstack(A_rows)
        e_theta = np.zeros(self.nv)
        e_theta[self.idx_theta] = 1.0
        rhs = -A @ e_theta

        mask = np.ones(self.nv, dtype=bool)
        mask[self.idx_theta] = False
        A_red = A[:, mask]
        x_red, *_ = np.linalg.lstsq(A_red, rhs, rcond=None)
        x = np.zeros(self.nv)
        x[mask] = x_red
        return e_theta + x

    def compute_tau_ext(self, q: np.ndarray) -> None:
        if (not self.p['wrench_enable']) or (self.fid_ext < 0):
            self.tau_ext[:] = 0.0
            self.tau_ext_theta = 0.0
            return
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(
            self.model, self.data, q, self.fid_ext, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        W = pin.Force(self.wrench_world)
        self.tau_ext = (J6.T @ W.vector).copy()
        self.tau_ext_theta = float(self.B.T @ self.tau_ext)

    def compute_passive_torques(self, q: np.ndarray, dq: np.ndarray) -> None:
        self.tau_pass[:] = 0.0
        self.tau_pass_theta = 0.0
        if not self.p['passive_enable']:
            return
        K_MAX = self.p['passive_K_MAX']
        exp_clip = self.p['passive_exp_clip']

        def safe_exp_stiffness(K0: float, alpha: float, error: float) -> float:
            x = alpha * abs(error)
            if x > exp_clip:
                return K_MAX
            return min(K0 * math.exp(x), K_MAX)

        idx_MCF = self.jid_MCF - 1
        idx_IFP = self.jid_IFP - 1

        q_MCF = float(q[idx_MCF])
        dq_MCF = float(dq[idx_MCF])
        err_MCF = q_MCF - self.p['rest_MCF']
        K_MCF = safe_exp_stiffness(self.p['K0_MCF'], self.p['alpha_MCF'], err_MCF)
        tau_MCF = -K_MCF * err_MCF - self.p['B_MCF'] * dq_MCF

        q_IFP = float(q[idx_IFP])
        dq_IFP = float(dq[idx_IFP])
        err_IFP = q_IFP - self.p['rest_IFP']
        K_IFP = safe_exp_stiffness(self.p['K0_IFP'], self.p['alpha_IFP'], err_IFP)
        tau_IFP = -K_IFP * err_IFP - self.p['B_IFP'] * dq_IFP

        self.tau_pass[idx_MCF] = tau_MCF
        self.tau_pass[idx_IFP] = tau_IFP
        self.tau_pass_theta = float(self.B.T @ self.tau_pass)

    def _stuck_try_release(self) -> bool:
        if not self.at_limit or (not self.have_valid):
            return False
        theta_dot = self.theta_dot
        b = self.p['fric_visc']
        fc = self.p['fric_coul']
        eps = self.p['fric_eps']
        tau_fric = b * theta_dot + fc * np.tanh(theta_dot / max(eps, 1e-9))
        tau_eff = (
            self.tau_m + self.tau_ext_theta + self.tau_pass_theta
            - tau_fric - self.gproj_last
        )
        if (tau_eff * self.limit_dir) >= -self.p['limit_release_tau']:
            return False
        base_step = self.p['limit_backoff_step']
        tries = self.p['limit_backoff_tries']
        for k in range(1, tries + 1):
            th_try = self.theta_valid - self.limit_dir * (base_step * k)
            th_try = self._clamp_theta_bounds(th_try)
            self.q = self.q_valid.copy()
            self.last_x = self.last_x_valid.copy()
            q_try, _ = self.solve_closure(th_try, update_warmstart=False)
            if q_try is None:
                continue
            self.theta = float(th_try)
            self.q = q_try
            self.B = self.compute_B(self.q)
            self.theta_dot = 0.0
            self.theta_ddot = 0.0
            self.dq[:] = 0.0
            self.ddq[:] = 0.0
            self.at_limit = False
            self._store_valid_state()
            return True
        return False

    def step(self, tau_m: float) -> Dict:
        """Single dynamics step. Returns dict with full state."""
        self.tau_m = tau_m
        if self.at_limit:
            released = self._stuck_try_release()
            if not released:
                self.dq[:] = 0.0
                self.ddq[:] = 0.0
                return self._make_output(0, 0, at_limit=1.0)

        self.theta = self._clamp_theta_bounds(self.theta)

        q_new, info = self.solve_closure(self.theta, update_warmstart=True)
        self._last_solver_success = bool(info['success'])
        self._last_solver_nfev = int(info['nfev'])
        self._last_closure_norm = float(info['closure_norm'])
        self._last_hit_bounds = bool(info['hit_bounds'])

        if q_new is None:
            self.at_limit = True
            self.limit_dir = float(np.sign(self.theta_dot)) if abs(self.theta_dot) > 1e-9 else 1.0
            self.theta = float(self.theta_valid)
            self.q = self.q_valid.copy()
            self.B = self.B_valid.copy()
            self.last_x = self.last_x_valid.copy()
            self.theta_dot = 0.0
            self.theta_ddot = 0.0
            self.dq[:] = 0.0
            self.ddq[:] = 0.0
            return self._make_output(self._last_closure_norm, self._last_solver_nfev, at_limit=1.0)

        self.q = q_new
        self.B_prev = self.B.copy()
        self.B = self.compute_B(self.q)
        self.Bdot = (self.B - self.B_prev) / self.dt

        self.at_limit = False
        self._store_valid_state()

        Bdot_local = self.Bdot
        self.dq = self.B * self.theta_dot
        self.ddq = self.B * self.theta_ddot + Bdot_local * self.theta_dot

        M = pin.crba(self.model, self.data, self.q)
        h = pin.nonLinearEffects(self.model, self.data, self.q, self.dq)

        self.compute_tau_ext(self.q)
        self.compute_passive_torques(self.q, self.dq)

        denom_mech = float(self.B.T @ (M @ self.B))
        Jm = self.p['motor_inertia']
        denom = denom_mech + Jm
        self.denom_last = denom

        self.gproj_last = float(self.B.T @ h)
        self.proj_last = float(self.B.T @ (M @ (Bdot_local * self.theta_dot) + h))

        if abs(denom) < self.p['denom_min']:
            self.at_limit = True
            self.limit_dir = float(np.sign(self.theta_dot)) if abs(self.theta_dot) > 1e-6 else 1.0
            self.theta = float(self.theta_valid)
            self.q = self.q_valid.copy()
            self.B = self.B_valid.copy()
            self.last_x = self.last_x_valid.copy()
            self.theta_dot = 0.0
            self.theta_ddot = 0.0
            self.dq[:] = 0.0
            self.ddq[:] = 0.0
            return self._make_output(self._last_closure_norm, self._last_solver_nfev, at_limit=1.0)

        b = self.p['fric_visc']
        fc = self.p['fric_coul']
        eps = self.p['fric_eps']
        tau_fric = b * self.theta_dot + fc * np.tanh(self.theta_dot / max(eps, 1e-9))
        tau_damp = self.p['damping_theta'] * self.theta_dot
        self.tau_fric_last = float(tau_fric)
        self.tau_damp_last = float(tau_damp)

        num = (self.tau_m + self.tau_pass_theta + self.tau_ext_theta) \
              - self.proj_last - tau_fric - tau_damp
        self.num_last = float(num)

        self.theta_ddot = num / denom
        max_dd = self.p['max_theta_ddot']
        self.theta_ddot = float(np.clip(self.theta_ddot, -max_dd, max_dd))

        self.theta_dot += self.theta_ddot * self.dt
        max_d = self.p['max_theta_dot']
        self.theta_dot = float(np.clip(self.theta_dot, -max_d, max_d))

        theta_next = self.theta + self.theta_dot * self.dt
        theta_next = self._clamp_theta_bounds(theta_next)

        if self.p['limit_hold']:
            if (abs(theta_next - self.theta) < 1e-12) and (abs(self.theta_dot) > 1e-8):
                self.at_limit = True
                self.limit_dir = (
                    float(np.sign(self.theta_dot)) if abs(self.theta_dot) > 1e-9 else 1.0
                )
                self.theta_dot = 0.0
                self.theta_ddot = 0.0
                self.theta = float(theta_next)
                self._store_valid_state()
                return self._make_output(self._last_closure_norm, self._last_solver_nfev, at_limit=1.0)

        self.theta = float(theta_next)
        self.dq = self.B * self.theta_dot
        self.ddq = self.B * self.theta_ddot + Bdot_local * self.theta_dot

        tau_rnea = pin.rnea(self.model, self.data, self.q, self.dq, self.ddq)
        self.tau_full = (tau_rnea - self.tau_ext - self.tau_pass).copy()

        return self._make_output(self._last_closure_norm, self._last_solver_nfev, at_limit=0.0)

    def _make_output(self, closure_norm, nfev, at_limit=0.0):
        """Build output dict with current state."""
        out = {
            'time': 0.0,  # will be filled by caller
            'theta': float(self.theta),
            'theta_dot': float(self.theta_dot),
            'theta_ddot': float(self.theta_ddot),
            'tau_m': float(self.tau_m),
            'M_eff': float(self.denom_last),
            'proj': float(self.proj_last),
            'g_proj': float(self.gproj_last),
            'tau_pass_theta': float(self.tau_pass_theta),
            'tau_ext_theta': float(self.tau_ext_theta),
            'tau_fric': float(self.tau_fric_last),
            'tau_damp': float(self.tau_damp_last),
            'closure_norm': float(closure_norm) if np.isfinite(closure_norm) else -1.0,
            'nfev': float(nfev),
            'at_limit': at_limit,
        }
        B = self.B
        Bdot = self.Bdot
        for j in range(self.nv):
            out[f'B_{j}'] = float(B[j])
        for j in range(self.nv):
            out[f'Bdot_{j}'] = float(Bdot[j])
        return out

    def run(self, n_steps: int, tau_profile: Callable[[float], float],
            progress_interval: int = 5000) -> List[Dict]:
        """Run n_steps with a time-based torque profile."""
        data: List[Dict] = []
        for i in range(n_steps):
            t = i * self.dt
            tau_m = tau_profile(t)
            result = self.step(tau_m)
            result['time'] = t
            result['step'] = i
            data.append(result)
            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                print(f"  step {i + 1:8d} / {n_steps}  |  "
                      f"theta = {self.theta:.4f} rad  |  "
                      f"at_limit = {self.at_limit}")
        return data


# ====================================================================
# TIME-BASED TORQUE PROFILE
# ====================================================================

def sine_time_profile(amplitude: float = 0.5, period: float = 10.0) -> Callable[[float], float]:
    """τ(t) = A * sin(2π * t / T) — same physical signal regardless of dt."""
    return lambda t: amplitude * math.sin(2.0 * math.pi * t / period)


def half_sine_neg_profile(amplitude: float = 3.5, period: float = 10.0) -> Callable[[float], float]:
    """τ(t) = -|A| * |sin(π * t / T)| — always non-positive, oscillates 0 → -|A| → 0.
    Drives the system in the negative-theta direction without ever pushing toward the
    positive kinematic limit, keeping motion well within the physical range."""
    return lambda t: -abs(amplitude) * abs(math.sin(math.pi * t / period))


# ====================================================================
# CSV FIELDS
# ====================================================================

def build_fieldnames(nv: int) -> List[str]:
    fields = [
        'step', 'time', 'theta', 'theta_dot', 'theta_ddot', 'tau_m',
        'M_eff', 'proj', 'g_proj',
        'tau_pass_theta', 'tau_ext_theta', 'tau_fric', 'tau_damp',
        'closure_norm', 'nfev', 'at_limit',
    ]
    for j in range(nv):
        fields.append(f'B_{j}')
    for j in range(nv):
        fields.append(f'Bdot_{j}')
    return fields


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate dt-sweep data for Bdot sensitivity analysis',
    )
    parser.add_argument('--urdf', type=str, default=None)
    parser.add_argument('--dt-list', type=str,
                        default='0.00005,0.0001,0.0005,0.001,0.002,0.005',
                        help='Comma-separated list of dt values (seconds)')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Simulation duration in seconds')
    parser.add_argument('--amplitude', type=float, default=None,
                        help='Sine torque amplitude (Nm). Default: 0.5, or 0.05 with --no-friction')
    parser.add_argument('--period', type=float, default=10.0,
                        help='Sine torque period (seconds)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--warmup', type=float, default=2.0,
                        help='Warmup duration in seconds (discarded from CSV)')
    parser.add_argument('--progress', type=int, default=5000)
    parser.add_argument('--no-friction', action='store_true',
                        help='Set fric_coul=fric_visc=damping_theta=0 to isolate the '
                             'Bdot FD approximation error from Euler stiffness effects. '
                             'Output goes to results/dt_sweep_no_friction/ by default.')
    parser.add_argument('--no-coulomb', action='store_true',
                        help='Set fric_coul=0 only, keeping fric_visc and damping_theta. '
                             'Removes stiff chattering while keeping linear damping for '
                             'bounded Euler integration. Best option to isolate Bdot error. '
                             'Output goes to results/dt_sweep_no_coulomb/ by default.')
    parser.add_argument('--theta-min', type=float, default=-2.5,
                        help='Physical lower bound on theta [rad] (default: -2.5). '
                             'Set to the actual kinematic limit, e.g. -0.75.')
    parser.add_argument('--theta-max', type=float, default=2.5,
                        help='Physical upper bound on theta [rad] (default: 2.5). '
                             'Set to the actual kinematic limit, e.g. 0.09.')
    parser.add_argument('--theta-init', type=float, default=0.0,
                        help='Initial theta position [rad] (default: 0.0).')
    parser.add_argument('--profile', type=str, default='sine',
                        choices=['sine', 'half_sine_neg'],
                        help='Torque profile: "sine" = A·sin(2πt/T) (default); '
                             '"half_sine_neg" = -|A|·|sin(πt/T)| always ≤ 0.')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.urdf is None:
        args.urdf = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'urdf', 'assembly_with_hand.urdf'))
    if args.no_friction and args.no_coulomb:
        print("ERROR: --no-friction and --no-coulomb are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    if args.amplitude is None:
        args.amplitude = 0.05 if args.no_friction else 0.5
    if args.output_dir is None:
        if args.no_friction:
            subdir = 'dt_sweep_no_friction'
        elif args.no_coulomb:
            subdir = 'dt_sweep_no_coulomb'
        else:
            subdir = 'dt_sweep'
        args.output_dir = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'results', subdir))
    if args.no_friction:
        friction_override = {'fric_coul': 0.0, 'fric_visc': 0.0, 'damping_theta': 0.0}
    elif args.no_coulomb:
        friction_override = {'fric_coul': 0.0}
    else:
        friction_override = {}

    # Physical theta bounds and initial position — always applied
    bounds_override = {
        'theta_min':  args.theta_min,
        'theta_max':  args.theta_max,
        'theta_init': args.theta_init,
    }
    param_override = {**friction_override, **bounds_override}

    os.makedirs(args.output_dir, exist_ok=True)

    dt_values = [float(x) for x in args.dt_list.split(',')]
    urdf_path = os.path.abspath(args.urdf)

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    # Load model once just to get nv for warmup
    model = pin.buildModelFromUrdf(urdf_path)
    nv = model.nv
    fieldnames = build_fieldnames(nv)

    # Save params
    params = {
        'urdf': urdf_path,
        'dt_list': dt_values,
        'duration': args.duration,
        'amplitude': args.amplitude,
        'period': args.period,
        'warmup': args.warmup,
        'nv': nv,
        'no_friction': args.no_friction,
        'no_coulomb': getattr(args, 'no_coulomb', False),
        'friction_override': friction_override,
        'theta_min':  args.theta_min,
        'theta_max':  args.theta_max,
        'theta_init': args.theta_init,
        'profile':    args.profile,
    }
    params_path = os.path.join(args.output_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Params saved: {params_path}")

    if args.profile == 'half_sine_neg':
        tau_fn = half_sine_neg_profile(amplitude=args.amplitude, period=args.period)
    else:
        tau_fn = sine_time_profile(amplitude=args.amplitude, period=args.period)

    for dt_val in dt_values:
        dt_ms = dt_val * 1000
        total_steps = int(args.duration / dt_val)
        warmup_steps = int(args.warmup / dt_val)

        print(f"\n{'=' * 60}")
        print(f"dt = {dt_val:.6f} s ({dt_ms:.3f} ms)  |  "
              f"steps = {total_steps}  |  sim = {args.duration} s")
        print(f"{'=' * 60}")

        t_start = time.time()
        bm = ExoDynamicsSweep(urdf_path, dt=dt_val, params=param_override)
        raw_data = bm.run(total_steps, tau_profile=tau_fn,
                          progress_interval=args.progress)
        t_elapsed = time.time() - t_start

        # Discard warmup
        data = raw_data[warmup_steps:]
        n_valid = len(data)
        sim_valid = n_valid * dt_val

        # Verify stability
        thetas = [d['theta'] for d in data]
        has_nan = any(math.isnan(t) or math.isinf(t) for t in thetas)
        stable = 'YES' if not has_nan else 'DIVERGED'

        print(f"  Wall clock: {t_elapsed:.2f}s  |  "
              f"valid steps: {n_valid}  |  "
              f"stable: {stable}")

        csv_path = os.path.join(args.output_dir, f'dt_{dt_val:.6f}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for d in data:
                writer.writerow({k: d.get(k, '') for k in fieldnames})
        print(f"  Saved: {csv_path}")

    print(f"\n{'=' * 60}")
    print("All simulations complete.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
