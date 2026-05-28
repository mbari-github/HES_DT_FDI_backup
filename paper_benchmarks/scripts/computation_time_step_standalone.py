#!/usr/bin/env python3
"""
Standalone computational cost benchmark for the HES reduced dynamics step().

Runs the core dynamics computation (Pinocchio + kinematic closure) WITHOUT any
ROS infrastructure, measuring pure computational cost of each phase of step().

Usage:
  python3 computation_time_step_standalone.py [options]

Examples:
  # Default 50000 steps with constant tau_m = 0.5 Nm
  python3 computation_time_step_standalone.py

  # 100k steps with sine torque profile
  python3 computation_time_step_standalone.py --steps 100000 --tau-profile sine

  # Custom URDF and output directory
  python3 computation_time_step_standalone.py --urdf /path/to/model.urdf --output-dir /tmp

Output:
  paper_benchmarks/results/step_standalone_timing.csv
  paper_benchmarks/results/step_standalone_statistics.txt
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

PUB_JOINT_NAMES = [
    "rev_crank", "rev_body2linkAC", "rev_crank2shaft", "slider",
    "rev_slider2linkBC", "rev_linkAC2linkCE",
    "rev_palmo2prossimale", "rev_prossimale2mediale", "slider2",
]


class ExoDynamicsBenchmark:
    """
    Standalone reduced dynamics (no ROS).

    Extracted from exoskeleton_dynamics/new_dynamics_with_hand.py and
    paper_benchmarks/paper_benchmarks/benchmark_dynamics.py, with all ROS
    dependencies removed. Runs the same core computation as the ROS node
    but without any middleware overhead.
    """

    def __init__(self, urdf_path: str, dt: float = 0.001, params: Optional[Dict] = None):
        self.dt = dt

        # Default parameters (from paper_benchmarks/config/dynamics_params.yaml)
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
            'theta_min': -0.75,
            'theta_max':  0.09,
            'theta_init': 0.0,
            'limit_hold': True,
            'limit_release_tau': 0.02,
            'limit_backoff_step': 0.003,
            'limit_backoff_tries': 6,
            'limit_use_theta_bounds': True,
            # External wrench (disabled by default in standalone)
            'wrench_enable': False,
            'wrench_frame': 'frame_CE_end_2',
            'wrench_scale': 1.0,
            # Passive finger model
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
            # Medial soft bound
            'medial_soft_enable': True,
            'medial_soft_lower': -2.2,
            'medial_soft_upper': 0.0,
            'medial_soft_weight': 1.0,
        }
        if params:
            self.p.update(params)

        # ---- Load Pinocchio model ----
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        # ---- Joint indices ----
        self.jid_crank = self.model.getJointId('rev_crank')
        self.idx_theta = self.jid_crank - 1

        self.jid_MCF = self.model.getJointId('rev_palmo2prossimale')
        self.jid_IFP = self.model.getJointId('rev_prossimale2mediale')

        # ---- Closure frame pairs ----
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

        # ---- Dependent DOFs (optimised by closure solver) ----
        dep_joint_names = [
            'rev_body2linkAC',
            'rev_crank2shaft',
            'slider',
            'rev_slider2linkBC',
            'rev_linkAC2linkCE',
            'rev_palmo2prossimale',
            'rev_prossimale2mediale',
            'slider2',
        ]
        dep_ids = [self.model.getJointId(n) for n in dep_joint_names]
        self.idx_opt = [jid - 1 for jid in dep_ids]

        self.lower = np.array(
            [-2.5, -2.5, -0.015, -2.5, -2.5, -1.5, -2.5, -0.008],
            dtype=float,
        )
        self.upper = np.array(
            [2.5, 2.5, 0.004, 2.5, 2.5, 1.2, 2.5, 0.008],
            dtype=float,
        )

        # ---- External wrench frame ----
        self.fid_ext = -1
        if self.p['wrench_enable']:
            self.fid_ext = self.model.getFrameId(self.p['wrench_frame'])
            if self.fid_ext < 0:
                self.p['wrench_enable'] = False

        # ---- Published joint index cache ----
        self.pub_joint_idxs = [self.model.getJointId(n) - 1 for n in PUB_JOINT_NAMES]

        # ---- Reduced state ----
        self.theta = float(self.p['theta_init'])
        self.theta_dot = 0.0
        self.theta_ddot = 0.0

        self.q = pin.neutral(self.model).copy()
        self.dq = np.zeros(self.nv)
        self.ddq = np.zeros(self.nv)

        self.last_x = np.zeros(len(self.idx_opt), dtype=float)
        self.B = np.zeros(self.nv)
        self.B_prev = np.zeros(self.nv)

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

        self.at_limit = False
        self.limit_dir = 0.0
        self.have_valid = False
        self.theta_valid = float(self.theta)
        self.q_valid = self.q.copy()
        self.dq_valid = self.dq.copy()
        self.B_valid = self.B.copy()
        self.last_x_valid = self.last_x.copy()

        # ---- Initial kinematic closure ----
        q0, info0 = self.solve_closure(self.theta, update_warmstart=True)
        if q0 is not None:
            self.q = q0
            self.B = self.compute_B(self.q)
            self._store_valid_state()
        else:
            self.at_limit = True
            self.theta_dot = 0.0
            self.theta_ddot = 0.0

    # ================================================================
    # HELPERS
    # ================================================================

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

    # ================================================================
    # KINEMATIC CLOSURE
    # ================================================================

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
            'success': False,
            'nfev': 0,
            'closure_norm': np.inf,
            'hit_bounds': False,
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
            residuals,
            self.last_x.copy(),
            method='trf',
            bounds=(self.lower, self.upper),
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
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

    # ================================================================
    # PROJECTION VECTOR B = dq/dtheta
    # ================================================================

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

    # ================================================================
    # EXTERNAL WRENCH
    # ================================================================

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

    # ================================================================
    # PASSIVE TORQUES (Fung exponential spring-damper)
    # ================================================================

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

    # ================================================================
    # DYNAMICS STEP (core computation with per-section timing)
    # ================================================================

    def step(self, tau_m: float, ext_wrench: Optional[np.ndarray] = None) -> Dict[str, float]:
        t0 = time.perf_counter_ns()

        if ext_wrench is not None:
            self.wrench_world = np.asarray(ext_wrench, dtype=float).ravel()
        else:
            self.wrench_world[:] = 0.0

        self.tau_m = tau_m

        # -- 0) Stuck-at-limit recovery ---------------------------------
        if self.at_limit:
            released = self._stuck_try_release()
            if not released:
                self.dq[:] = 0.0
                self.ddq[:] = 0.0
                return {
                    'total_ns': time.perf_counter_ns() - t0,
                    'closure_ns': 0,
                    'compute_B_ns': 0,
                    'crba_ns': 0,
                    'ext_passive_ns': 0,
                    'rnea_ns': 0,
                    'closure_norm': self.last_closure_norm if hasattr(self, 'last_closure_norm') else -1.0,
                    'nfev': 0,
                    'at_limit': 1.0,
                }

        # -- 1) Clamp theta --------------------------------------------
        self.theta = self._clamp_theta_bounds(self.theta)

        # -- 2) Kinematic closure --------------------------------------
        tc0 = time.perf_counter_ns()
        q_new, info = self.solve_closure(self.theta, update_warmstart=True)
        tc1 = time.perf_counter_ns()
        closure_ns = tc1 - tc0

        self.last_solver_success = bool(info['success'])
        self.last_solver_nfev = int(info['nfev'])
        self.last_closure_norm = float(info['closure_norm'])
        self.last_hit_bounds = bool(info['hit_bounds'])

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
            return {
                'total_ns': time.perf_counter_ns() - t0,
                'closure_ns': closure_ns,
                'compute_B_ns': 0,
                'crba_ns': 0,
                'ext_passive_ns': 0,
                'rnea_ns': 0,
                'closure_norm': self.last_closure_norm,
                'nfev': self.last_solver_nfev,
                'at_limit': 1.0,
            }

        # -- 3) Update kinematics + compute B --------------------------
        self.q = q_new
        self.B_prev = self.B.copy()
        tb0 = time.perf_counter_ns()
        self.B = self.compute_B(self.q)
        tb1 = time.perf_counter_ns()
        compute_B_ns = tb1 - tb0

        self.at_limit = False
        self._store_valid_state()

        Bdot = (self.B - self.B_prev) / self.dt

        # -- 4) Preliminary dq, ddq ------------------------------------
        self.dq = self.B * self.theta_dot
        self.ddq = self.B * self.theta_ddot + Bdot * self.theta_dot

        # -- 5) CRBA + nonLinearEffects --------------------------------
        tcba0 = time.perf_counter_ns()
        M = pin.crba(self.model, self.data, self.q)
        h = pin.nonLinearEffects(self.model, self.data, self.q, self.dq)
        tcba1 = time.perf_counter_ns()
        crba_ns = tcba1 - tcba0

        # -- 6) External wrench + passive torques ----------------------
        tep0 = time.perf_counter_ns()
        self.compute_tau_ext(self.q)
        self.compute_passive_torques(self.q, self.dq)
        tep1 = time.perf_counter_ns()
        ext_passive_ns = tep1 - tep0

        # -- 7) Reduced scalar quantities ------------------------------
        denom_mech = float(self.B.T @ (M @ self.B))
        Jm = self.p['motor_inertia']
        denom = denom_mech + Jm
        self.denom_last = denom

        self.gproj_last = float(self.B.T @ h)
        self.proj_last = float(self.B.T @ (M @ (Bdot * self.theta_dot) + h))

        denom_min = self.p['denom_min']
        if abs(denom) < denom_min:
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
            return {
                'total_ns': time.perf_counter_ns() - t0,
                'closure_ns': closure_ns,
                'compute_B_ns': compute_B_ns,
                'crba_ns': crba_ns,
                'ext_passive_ns': ext_passive_ns,
                'rnea_ns': 0,
                'closure_norm': self.last_closure_norm,
                'nfev': self.last_solver_nfev,
                'at_limit': 1.0,
            }

        # -- 8) Friction + damping -------------------------------------
        b = self.p['fric_visc']
        fc = self.p['fric_coul']
        eps = self.p['fric_eps']
        tau_fric = b * self.theta_dot + fc * np.tanh(self.theta_dot / max(eps, 1e-9))
        tau_damp = self.p['damping_theta'] * self.theta_dot

        # -- 9) Reduced equation of motion -----------------------------
        num = (self.tau_m + self.tau_pass_theta + self.tau_ext_theta) \
              - self.proj_last - tau_fric - tau_damp

        self.theta_ddot = num / denom
        max_dd = self.p['max_theta_ddot']
        self.theta_ddot = float(np.clip(self.theta_ddot, -max_dd, max_dd))

        # -- 10) Explicit Euler integration ----------------------------
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
                return {
                    'total_ns': time.perf_counter_ns() - t0,
                    'closure_ns': closure_ns,
                    'compute_B_ns': compute_B_ns,
                    'crba_ns': crba_ns,
                    'ext_passive_ns': ext_passive_ns,
                    'rnea_ns': 0,
                    'closure_norm': self.last_closure_norm,
                    'nfev': self.last_solver_nfev,
                    'at_limit': 1.0,
                }

        self.theta = float(theta_next)

        # -- 11) Final dq, ddq -----------------------------------------
        self.dq = self.B * self.theta_dot
        self.ddq = self.B * self.theta_ddot + Bdot * self.theta_dot

        # -- 12) RNEA --------------------------------------------------
        tr0 = time.perf_counter_ns()
        tau_rnea = pin.rnea(self.model, self.data, self.q, self.dq, self.ddq)
        tr1 = time.perf_counter_ns()
        rnea_ns = tr1 - tr0

        self.tau_full = (tau_rnea - self.tau_ext - self.tau_pass).copy()

        t1 = time.perf_counter_ns()

        return {
            'total_ns': t1 - t0,
            'closure_ns': closure_ns,
            'compute_B_ns': compute_B_ns,
            'crba_ns': crba_ns,
            'ext_passive_ns': ext_passive_ns,
            'rnea_ns': rnea_ns,
            'closure_norm': self.last_closure_norm,
            'nfev': self.last_solver_nfev,
            'at_limit': 0.0,
            'theta': float(self.theta),
            'theta_dot': float(self.theta_dot),
        }

    # ================================================================
    # STUCK RELEASE
    # ================================================================

    def _stuck_try_release(self) -> bool:
        if not self.at_limit or (not self.have_valid):
            return False

        theta_dot = self.theta_dot
        b = self.p['fric_visc']
        fc = self.p['fric_coul']
        eps = self.p['fric_eps']
        tau_fric = b * theta_dot + fc * np.tanh(theta_dot / max(eps, 1e-9))

        tau_eff = (
            self.tau_m
            + self.tau_ext_theta
            + self.tau_pass_theta
            - tau_fric
            - self.gproj_last
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

    # ================================================================
    # RAMP STEP (prescribed theta, no integration)
    # ================================================================

    def step_ramp(self, meas_step: int, n_half: int) -> Dict:
        """Prescribe theta as triangular wave; run all sections timed."""
        t0 = time.perf_counter_ns()

        ramp_v = (self.p['theta_max'] - self.p['theta_min']) / (n_half * self.dt)
        half_idx = meas_step % (2 * n_half)
        if half_idx < n_half:
            self.theta = self.p['theta_min'] + ramp_v * half_idx * self.dt
            self.theta_dot = ramp_v
        else:
            self.theta = self.p['theta_max'] - ramp_v * (half_idx - n_half) * self.dt
            self.theta_dot = -ramp_v
        self.theta_ddot = 0.0

        # 1) Closure
        tc0 = time.perf_counter_ns()
        q_new, info = self.solve_closure(self.theta, update_warmstart=True)
        closure_ns = time.perf_counter_ns() - tc0

        def _joints_zero() -> Dict:
            return {n: 0.0 for n in PUB_JOINT_NAMES}

        if q_new is None:
            return {
                'total_ns': time.perf_counter_ns() - t0,
                'closure_ns': closure_ns, 'compute_B_ns': 0, 'crba_ns': 0,
                'ext_passive_ns': 0, 'rnea_ns': 0,
                'closure_norm': float(info['closure_norm']), 'nfev': int(info['nfev']),
                'at_limit': 1.0, 'theta': float(self.theta), 'theta_dot': float(self.theta_dot),
                **_joints_zero(),
            }

        # 2) Update q, compute B
        self.q = q_new
        self.B_prev = self.B.copy()
        tb0 = time.perf_counter_ns()
        self.B = self.compute_B(self.q)
        compute_B_ns = time.perf_counter_ns() - tb0

        Bdot = (self.B - self.B_prev) / self.dt
        self.dq = self.B * self.theta_dot
        self.ddq = Bdot * self.theta_dot  # theta_ddot=0

        # 3) CRBA + nonLinearEffects
        tcba0 = time.perf_counter_ns()
        M = pin.crba(self.model, self.data, self.q)
        h = pin.nonLinearEffects(self.model, self.data, self.q, self.dq)
        crba_ns = time.perf_counter_ns() - tcba0

        # 4) External wrench + passive
        tep0 = time.perf_counter_ns()
        self.tau_ext[:] = 0.0
        self.tau_ext_theta = 0.0
        self.compute_passive_torques(self.q, self.dq)
        ext_passive_ns = time.perf_counter_ns() - tep0

        # 5) RNEA
        tr0 = time.perf_counter_ns()
        tau_rnea = pin.rnea(self.model, self.data, self.q, self.dq, self.ddq)
        rnea_ns = time.perf_counter_ns() - tr0
        self.tau_full = (tau_rnea - self.tau_ext - self.tau_pass).copy()

        t1 = time.perf_counter_ns()

        result: Dict = {
            'total_ns': t1 - t0,
            'closure_ns': closure_ns, 'compute_B_ns': compute_B_ns,
            'crba_ns': crba_ns, 'ext_passive_ns': ext_passive_ns, 'rnea_ns': rnea_ns,
            'closure_norm': float(info['closure_norm']), 'nfev': int(info['nfev']),
            'at_limit': 0.0, 'theta': float(self.theta), 'theta_dot': float(self.theta_dot),
        }
        for name, idx in zip(PUB_JOINT_NAMES, self.pub_joint_idxs):
            result[name] = float(self.q[idx])
        return result

    # ================================================================
    # BENCHMARK LOOP
    # ================================================================

    def run(
        self,
        n_steps: int,
        tau_profile: Callable[[int], float],
        wrench_profile: Optional[Callable[[int], np.ndarray]] = None,
        progress_interval: int = 5000,
    ) -> List[Dict]:
        data: List[Dict] = []

        for i in range(n_steps):
            tau_m = tau_profile(i)
            ext_wrench = wrench_profile(i) if wrench_profile is not None else None
            result = self.step(tau_m, ext_wrench)
            result['step'] = i
            data.append(result)

            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                avg_ms = np.mean([d['total_ns'] for d in data[-progress_interval:]]) / 1e6
                print(f"  step {i + 1:8d} / {n_steps}  |  "
                      f"avg step = {avg_ms:.3f} ms  |  "
                      f"theta = {self.theta:.4f} rad  |  "
                      f"at_limit = {self.at_limit}")

        return data


# ====================================================================
# TORQUE PROFILES
# ====================================================================

def constant_profile(tau_amplitude: float = 0.5, **kwargs) -> Callable[[int], float]:
    return lambda i: tau_amplitude


def step_profile(
    tau_amplitude: float = 0.5,
    ramp_steps: int = 500,
    hold_steps: int = 5000,
    pause_steps: int = 2000,
) -> Callable[[int], float]:
    def profile(i: int) -> float:
        cycle = ramp_steps + hold_steps + ramp_steps + pause_steps
        pos = i % cycle
        if pos < ramp_steps:
            return tau_amplitude * pos / ramp_steps
        elif pos < ramp_steps + hold_steps:
            return tau_amplitude
        elif pos < ramp_steps + hold_steps + ramp_steps:
            t = pos - ramp_steps - hold_steps
            return tau_amplitude * (1.0 - t / ramp_steps)
        else:
            return 0.0
    return profile


def sine_profile(
    tau_amplitude: float = 0.5,
    period_steps: int = 10000,
) -> Callable[[int], float]:
    return lambda i: tau_amplitude * math.sin(2.0 * math.pi * i / period_steps)


def random_profile(
    tau_amplitude: float = 0.5,
    seed: int = 42,
) -> Callable[[int], float]:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-tau_amplitude, tau_amplitude, size=10000)
    return lambda i: float(values[i % len(values)])


# ====================================================================
# PROFILES LOOKUP
# ====================================================================

PROFILES = {
    'constant': constant_profile,
    'step': step_profile,
    'sine': sine_profile,
    'random': random_profile,
}


# ====================================================================
# STATISTICS
# ====================================================================

def compute_statistics(data: List[Dict], fields: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for field in fields:
        values = np.array([d[field] for d in data], dtype=float)
        stats[field] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99)),
            'max': float(np.max(values)),
        }
    return stats


def print_statistics(stats: Dict[str, Dict[str, float]], title: str = "=== Statistics ===") -> str:
    lines = [title]
    for field, s in stats.items():
        lines.append(f"\n{field}:")
        lines.append(f"  Mean:   {s['mean']:.4f}")
        lines.append(f"  Std:    {s['std']:.4f}")
        lines.append(f"  Min:    {s['min']:.4f}")
        lines.append(f"  Median: {s['median']:.4f}")
        lines.append(f"  P95:    {s['p95']:.4f}")
        lines.append(f"  P99:    {s['p99']:.4f}")
        lines.append(f"  Max:    {s['max']:.4f}")
    return '\n'.join(lines)


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Standalone computational cost benchmark for HES dynamics step()',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 50000 steps, sine torque at 0.5 Nm
  python3 computation_time_step_standalone.py

  # Constant torque, 100k steps, detailed section breakdown
  python3 computation_time_step_standalone.py --steps 100000 --tau-profile constant --tau-amplitude 0.3

  # Step (trapezoidal) torque profile
  python3 computation_time_step_standalone.py --tau-profile step --tau-amplitude 0.6

  # Custom URDF
  python3 computation_time_step_standalone.py --urdf /path/to/model.urdf
        """,
    )
    parser.add_argument('--urdf', type=str, default=None,
                        help='Path to URDF file (default: ../urdf/assembly_with_hand.urdf)')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Number of dynamics steps to run (default: 50000)')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='Integration time step in seconds (default: 0.001)')
    parser.add_argument('--warmup', type=int, default=2000,
                        help='Warmup steps discarded from statistics (default: 2000)')
    parser.add_argument('--tau-profile', type=str, default='sine',
                        choices=list(PROFILES.keys()),
                        help='Torque profile type (default: sine)')
    parser.add_argument('--tau-amplitude', type=float, default=0.5,
                        help='Torque amplitude in Nm (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for CSV and statistics (default: ../results/)')
    parser.add_argument('--progress', type=int, default=5000,
                        help='Progress reporting interval in steps (default: 5000, 0=silent)')

    args = parser.parse_args()

    # ---- Resolve paths ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.urdf is None:
        args.urdf = os.path.normpath(os.path.join(script_dir, '..', 'urdf', 'assembly_with_hand.urdf'))
    if args.output_dir is None:
        args.output_dir = os.path.normpath(os.path.join(script_dir, '..', 'results'))

    os.makedirs(args.output_dir, exist_ok=True)

    urdf_path = os.path.abspath(args.urdf)
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF file not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Print configuration ----
    n_half = args.steps // 4
    _theta_span = 0.09 - (-0.75)  # theta_max - theta_min
    ramp_v = _theta_span / (n_half * args.dt)
    print("=" * 65)
    print("HES Dynamics — Standalone Computational Cost Benchmark")
    print("=" * 65)
    print(f"  URDF:         {urdf_path}")
    print(f"  dt:           {args.dt} s")
    print(f"  Steps:        {args.steps}  (warmup: {args.warmup})")
    print(f"  Profile:      RAMP  theta=[-0.75, 0.09] rad  n_half={n_half}  ramp_v={ramp_v:.3f} rad/s")
    print(f"  Output dir:   {args.output_dir}")
    print("-" * 65)

    # ---- Instantiate benchmark ----
    print("\nInitialising dynamics model...")
    t_init = time.time()
    bm = ExoDynamicsBenchmark(urdf_path, dt=args.dt)
    t_init = time.time() - t_init
    print(f"  Model loaded ({bm.model.name}): nq={bm.nq}, nv={bm.nv}")
    print(f"  Closure frame pairs: {len(bm.closure_frame_pairs)}")
    print(f"  Dependent DOFs: {len(bm.idx_opt)}")
    print(f"  Initialisation time: {t_init:.3f} s")
    print(f"  Initial theta: {bm.theta:.4f} rad")

    # ---- Run benchmark (ramp profile) ----
    total_steps = args.steps + args.warmup
    sim_time = total_steps * args.dt
    print(f"\nRunning {total_steps} steps (simulated time: {sim_time:.1f} s)...")
    t_run_start = time.time()

    raw_data: List[Dict] = []
    for i in range(total_steps):
        meas_step = max(0, i - args.warmup)
        result = bm.step_ramp(meas_step, n_half)
        result['step'] = i - args.warmup
        raw_data.append(result)
        if args.progress > 0 and (i + 1) % args.progress == 0:
            avg_ms = np.mean([d['total_ns'] for d in raw_data[-args.progress:]]) / 1e6
            print(f"  step {i + 1:8d} / {total_steps}  |  avg={avg_ms:.3f} ms  |  theta={bm.theta:.4f} rad")

    t_run = time.time() - t_run_start

    # ---- Discard warmup ----
    data = raw_data[args.warmup:]
    n_valid = len(data)
    sim_time_valid = n_valid * args.dt

    # ---- Compute real-time factor ----
    rt_factor = t_run / sim_time if sim_time > 0 else 0.0
    rt_factor_valid = t_run / sim_time_valid if sim_time_valid > 0 else 0.0

    print(f"\n  Wall clock:    {t_run:.2f} s")
    print(f"  Simulated:     {sim_time_valid:.1f} s (valid) / {sim_time:.1f} s (total)")
    print(f"  RT factor:     {rt_factor_valid:.2f}x  (1.0 = real-time)")
    print(f"  Effective freq: {n_valid / t_run:.0f} step/s")

    # ---- Compute statistics ----
    fields_ms = ['total_ns', 'closure_ns', 'compute_B_ns', 'crba_ns', 'ext_passive_ns', 'rnea_ns']
    stats_ns = compute_statistics(data, fields_ms)

    # Convert ns → ms for display
    stats_ms = {}
    for field, s in stats_ns.items():
        name = field.replace('_ns', '_ms')
        stats_ms[name] = {k: v / 1e6 for k, v in s.items()}

    # Add percentage breakdown
    total_mean_ns = stats_ns['total_ns']['mean']
    sections = ['closure_ns', 'compute_B_ns', 'crba_ns', 'ext_passive_ns', 'rnea_ns']
    breakdown = {}
    other_ns = total_mean_ns - sum(stats_ns[s]['mean'] for s in sections)
    for s in sections:
        name = s.replace('_ns', '')
        pct = (stats_ns[s]['mean'] / total_mean_ns) * 100
        breakdown[name] = {
            'mean_ms': stats_ns[s]['mean'] / 1e6,
            'pct': pct,
        }
    breakdown['other'] = {
        'mean_ms': other_ns / 1e6,
        'pct': (other_ns / total_mean_ns) * 100,
    }

    # ---- Print timing statistics ----
    print("\n--- Section Timing (ms) ---")
    print(f"{'Section':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print("-" * 84)
    for field, s in stats_ms.items():
        print(f"{field:<20} {s['mean']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} "
              f"{s['median']:>8.4f} {s['p95']:>8.4f} {s['p99']:>8.4f} {s['max']:>8.4f}")

    # Also print the basic total_ms
    total_ms = np.array([d['total_ns'] for d in data], dtype=float) / 1e6
    print(f"\n{'total_ms':<20} {np.mean(total_ms):>8.4f} {np.std(total_ms):>8.4f} {np.min(total_ms):>8.4f} "
          f"{np.median(total_ms):>8.4f} {np.percentile(total_ms, 95):>8.4f} "
          f"{np.percentile(total_ms, 99):>8.4f} {np.max(total_ms):>8.4f}")

    print("\n--- Section Breakdown ---")
    print(f"{'Section':<20} {'Mean (ms)':>10} {'% of total':>10}")
    print("-" * 42)
    for name, bd in breakdown.items():
        print(f"{name:<20} {bd['mean_ms']:>10.4f} {bd['pct']:>9.1f}%")

    # ---- Closure statistics ----
    closure_norm = np.array([d['closure_norm'] for d in data], dtype=float)
    nfev = np.array([d['nfev'] for d in data], dtype=float)
    at_limit_count = sum(1 for d in data if d.get('at_limit', 0) > 0.5)

    print(f"\n--- Solver Statistics ---")
    print(f"  Closure norm: mean = {np.mean(closure_norm):.3e}, max = {np.max(closure_norm):.3e}")
    print(f"  NFEV:          mean = {np.mean(nfev):.1f}, max = {np.max(nfev):.0f}")
    print(f"  at_limit:     {at_limit_count} / {n_valid} steps ({100*at_limit_count/n_valid:.1f}%)")

    # ---- Save CSV ----
    csv_fields = ['step', 'total_ns', 'closure_ns', 'compute_B_ns', 'crba_ns',
                  'ext_passive_ns', 'rnea_ns', 'closure_norm', 'nfev', 'at_limit',
                  'theta', 'theta_dot'] + PUB_JOINT_NAMES
    csv_path = os.path.join(args.output_dir, 'step_standalone_timing.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for d in data:
            writer.writerow({k: d.get(k, '') for k in csv_fields})
    print(f"\nSaved: {csv_path}")

    # ---- Save statistics ----
    stats_lines = [
        "=" * 55,
        "HES Dynamics — Standalone Benchmark Statistics",
        "=" * 55,
        f"URDF:           {urdf_path}",
        f"dt:             {args.dt} s",
        f"Total steps:    {n_valid}",
        f"Wall clock:     {t_run:.2f} s",
        f"Simulated:      {sim_time_valid:.1f} s",
        f"RT factor:      {rt_factor_valid:.2f}x",
        f"Profile:        RAMP  theta=[-0.75, 0.09] rad  n_half={n_half}  ramp_v={ramp_v:.3f} rad/s",
        "",
        "--- Timing (ms) ---",
    ]
    for field, s in stats_ms.items():
        stats_lines.append(f"{field}:")
        for k, v in s.items():
            stats_lines.append(f"  {k}: {v:.4f}")
        stats_lines.append("")

    total_ms_arr = np.array([d['total_ns'] for d in data], dtype=float) / 1e6
    stats_lines.append("total_ms:")
    for name, fn in [('mean', np.mean), ('std', np.std), ('min', np.min),
                      ('median', np.median), ('p95', lambda x: np.percentile(x, 95)),
                      ('p99', lambda x: np.percentile(x, 99)), ('max', np.max)]:
        stats_lines.append(f"  {name}: {fn(total_ms_arr):.4f}")
    stats_lines.append("")

    stats_lines.append("--- Section Breakdown ---")
    for name, bd in breakdown.items():
        stats_lines.append(f"  {name}: {bd['mean_ms']:.4f} ms ({bd['pct']:.1f}%)")
    stats_lines.append("")

    stats_lines.append("--- Solver ---")
    stats_lines.append(f"  closure_norm mean: {np.mean(closure_norm):.3e}")
    stats_lines.append(f"  closure_norm max:  {np.max(closure_norm):.3e}")
    stats_lines.append(f"  nfev mean:         {np.mean(nfev):.1f}")
    stats_lines.append(f"  nfev max:          {np.max(nfev):.0f}")
    stats_lines.append(f"  at_limit:          {at_limit_count} / {n_valid}")

    stats_text = '\n'.join(stats_lines)

    stats_path = os.path.join(args.output_dir, 'step_standalone_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write(stats_text + '\n')
    print(f"Saved: {stats_path}")

    # ---- Summary line ----
    print("\n" + "=" * 65)
    print(f"  Mean step time: {np.mean(total_ms):.4f} ms  |  "
          f"P95: {np.percentile(total_ms, 95):.4f} ms  |  "
          f"RT factor: {rt_factor_valid:.2f}x")
    print("=" * 65)


if __name__ == '__main__':
    main()
