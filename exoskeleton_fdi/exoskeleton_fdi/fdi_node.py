#!/usr/bin/env python3
"""
fdi_node.py — Standalone Fault Detection and Isolation for Exoskeleton Sensors
================================================================================

ROS 2 node that detects and isolates sensor faults on two channels:

  Channel 0 — tau_ext_theta (load cell / external force sensor)
  Channel 3 — joint_states (encoder position feedback)

Architecture
------------
Two dedicated residual generators with (ideally) diagonal signature:

  R_force (ch0) — Admittance inversion
    Reconstructs tau_ext from the trajectory reference by inverting the
    admittance controller model. Uses ONLY the reference (no plant feedback),
    so it is sensitive to tau_ext faults but immune to encoder faults.

    tau_hat_ext = Mv * theta_ddot_ref + Dv * theta_dot_ref +
                  Kv * (theta_ref - theta_eq) - tau_hat_wall

    r_force = tau_ext_meas - tau_hat_ext

    The residual is zeroed and force_valid=False when the controller is in
    hard saturation or the measured tau_ext is within the deadband.

  R_encoder (ch3) — Luenberger state observer
    Predicts theta_hat and theta_dot_hat using the plant model driven by
    tau_m, tau_ext_meas, and feed-forward terms. Compares prediction with
    the measured encoder position.

    theta_ddot_model = (tau_m + tau_pass + tau_ext_meas - proj - tau_friction) / M_eff
    theta_dot_hat += dt * (theta_ddot_model + l2 * (theta_meas - theta_hat))
    theta_hat     += dt * (theta_dot_hat + l1 * (theta_meas - theta_hat))

    r_encoder = theta_meas - theta_hat

    If tau_ext_meas is corrupted (ch0 fault), both the plant and the observer
    see the same corrupted value, so the residual remains small. If the encoder
    is corrupted (ch3 fault), the prediction is based on clean inputs but the
    measurement is wrong, so the residual grows.

Diagnosis
---------
  r_force_filt   = EMA(|r_force|, alpha)     [only when force_valid]
  r_encoder_filt = EMA(|r_encoder|, alpha)

  Fault is confirmed after debounce_count consecutive samples above threshold.
  Output: HEALTHY | FAULT_CH0 | FAULT_CH3

Subscribed topics
-----------------
  /joint_states                       sensor_msgs/JointState
  /exo_dynamics/tau_ext_theta         exoskeleton_safety_msgs/Float64Stamped
  /torque_raw (remapped to /torque)   exoskeleton_safety_msgs/Float64Stamped
  /trajectory_ref                     exoskeleton_safety_msgs/Float64ArrayStamped
  /exo_dynamics/ff_terms              exoskeleton_safety_msgs/Float64ArrayStamped

Published topics
----------------
  /fdi/residuals    std_msgs/Float64MultiArray        [r_force, r_encoder]
  /fdi/diagnosis    std_msgs/String                   HEALTHY | FAULT_CH0 | FAULT_CH3
  /fdi/debug        exoskeleton_safety_msgs/Float64ArrayStamped
      Layout (12 fields):
        [0]  theta_meas
        [1]  theta_hat
        [2]  theta_dot_meas
        [3]  theta_dot_hat
        [4]  tau_ext_meas
        [5]  tau_ext_hat (0.0 if force_valid=False)
        [6]  r_force
        [7]  r_encoder
        [8]  r_force_filtered
        [9]  r_encoder_filtered
        [10] force_valid (1.0 / 0.0)
        [11] diagnosis_id (0=healthy, 1=ch0, 2=ch3)

ROS 2 Parameters
----------------
  joint_name             (str)   Joint to observe                default: 'rev_crank'
  publish_rate           (float) Loop frequency [Hz]             default: 200.0

  # Plant model (must match dynamics_params.yaml)
  fric_visc              (float) Viscous friction [Nm*s/rad]     default: 2.0
  fric_coul              (float) Coulomb friction [Nm]           default: 2.0
  fric_eps               (float) tanh Coulomb threshold [rad/s]  default: 0.005
  damping_theta          (float) Additional damping [Nm*s/rad]   default: 1.0

  # Admittance params (must match admittance_controller)
  adm_M_virt             (float) Virtual mass                    default: 0.5
  adm_D_virt             (float) Virtual damping                 default: 5.0
  adm_K_virt             (float) Virtual stiffness               default: 2.0
  adm_theta_eq           (float) Equilibrium position            default: 0.0
  adm_force_deadband     (float) Force deadband [Nm]             default: 0.0
  adm_theta_ref_min      (float) Lower theta_ref limit [rad]     default: -0.75
  adm_theta_ref_max      (float) Upper theta_ref limit [rad]     default: 0.09
  adm_wall_buffer        (float) Virtual wall buffer [rad]       default: 0.05
  adm_K_wall             (float) Wall stiffness [Nm/rad]         default: 80.0
  adm_D_wall             (float) Wall damping [Nm*s/rad]         default: 10.0

  # Luenberger poles
  enc_obs_pole_1         (float) Pole 1 (negative, stable)       default: -15.0
  enc_obs_pole_2         (float) Pole 2 (negative, stable)       default: -20.0

  # Detection thresholds
  thresh_force           (float) Force residual threshold [Nm]   default: 0.5
  thresh_encoder         (float) Encoder residual threshold [rad]default: 0.02

  # Debounce
  debounce_count         (int)   Consecutive samples for fault   default: 10

  # EMA filter
  residual_alpha         (float) EMA coefficient [0, 1]          default: 0.1
"""

import math

import rclpy
from rclpy.node import Node

from exoskeleton_safety_msgs.msg import Float64Stamped, Float64ArrayStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def ema(prev: float, new_val: float, alpha: float) -> float:
    """Exponential moving average."""
    return alpha * new_val + (1.0 - alpha) * prev


class Diagnosis:
    HEALTHY = 0
    FAULT_CH0 = 1
    FAULT_CH3 = 2


DIAG_NAMES = {
    Diagnosis.HEALTHY: 'HEALTHY',
    Diagnosis.FAULT_CH0: 'FAULT_CH0',
    Diagnosis.FAULT_CH3: 'FAULT_CH3',
}


# ─────────────────────────────────────────────────────────────
# FDI Node
# ─────────────────────────────────────────────────────────────

class FDINode(Node):

    def __init__(self):
        super().__init__('fdi_node')

        # ── Parameters ──────────────────────────────────────
        self.declare_parameter('joint_name', 'rev_crank')
        self.declare_parameter('publish_rate', 200.0)

        # Plant model
        self.declare_parameter('fric_visc', 2.0)
        self.declare_parameter('fric_coul', 2.0)
        self.declare_parameter('fric_eps', 0.005)
        self.declare_parameter('damping_theta', 1.0)

        # Admittance inversion
        self.declare_parameter('adm_M_virt', 0.5)
        self.declare_parameter('adm_D_virt', 5.0)
        self.declare_parameter('adm_K_virt', 2.0)
        self.declare_parameter('adm_theta_eq', 0.0)
        self.declare_parameter('adm_force_deadband', 0.0)
        self.declare_parameter('adm_theta_ref_min', -0.75)
        self.declare_parameter('adm_theta_ref_max', 0.09)
        self.declare_parameter('adm_wall_buffer', 0.05)
        self.declare_parameter('adm_K_wall', 80.0)
        self.declare_parameter('adm_D_wall', 10.0)

        # Luenberger poles
        self.declare_parameter('enc_obs_pole_1', -15.0)
        self.declare_parameter('enc_obs_pole_2', -20.0)

        # Detection thresholds
        self.declare_parameter('thresh_force', 0.5)
        self.declare_parameter('thresh_encoder', 0.02)

        # Debounce
        self.declare_parameter('debounce_count', 10)

        # EMA filter
        self.declare_parameter('residual_alpha', 0.1)

        # Hold timer (prevents cross-triggering oscillation)
        self.declare_parameter('hold_time_ms', 500)

        rate = self.get_parameter('publish_rate').value
        self._dt = 1.0 / rate
        self._joint_name = self.get_parameter('joint_name').value

        # ── Input signals ───────────────────────────────────
        self._theta_meas = 0.0
        self._theta_dot_meas = 0.0
        self._tau_ext_meas = 0.0
        self._tau_m = 0.0

        self._M_eff = 1.0
        self._proj = 0.0
        self._tau_pass = 0.0

        self._theta_ref = 0.0
        self._theta_dot_ref = 0.0
        self._theta_ddot_ref = 0.0

        # Reception flags
        self._js_received = False
        self._tau_ext_received = False
        self._tau_m_received = False
        self._ff_received = False
        self._traj_received = False

        # ── R_force state ───────────────────────────────────
        self._tau_ext_hat = 0.0
        self._tau_wall_hat = 0.0
        self._force_valid = False
        self._r_force = 0.0

        # ── R_encoder state ─────────────────────────────────
        self._theta_hat = 0.0
        self._theta_dot_hat = 0.0
        self._enc_obs_initialized = False
        self._r_encoder = 0.0

        # ── Filtered residuals ──────────────────────────────
        self._r_force_filtered = 0.0
        self._r_encoder_filtered = 0.0

        # ── Debounce state ──────────────────────────────────
        self._debounce_force = 0
        self._debounce_encoder = 0
        self._diagnosis = Diagnosis.HEALTHY
        self._hold_steps = 0

        # ── Subscribers ─────────────────────────────────────
        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10
        )
        self.create_subscription(
            Float64Stamped, '/exo_dynamics/tau_ext_theta',
            self._tau_ext_cb, 10
        )
        self.create_subscription(
            Float64Stamped, '/torque_raw', self._tau_m_cb, 10
        )
        self.create_subscription(
            Float64ArrayStamped, '/exo_dynamics/ff_terms',
            self._ff_cb, 10
        )
        self.create_subscription(
            Float64ArrayStamped, '/trajectory_ref',
            self._traj_cb, 10
        )

        # ── Publishers ──────────────────────────────────────
        self._pub_residuals = self.create_publisher(
            Float64MultiArray, '/fdi/residuals', 10
        )
        self._pub_diagnosis = self.create_publisher(
            String, '/fdi/diagnosis', 10
        )
        self._pub_debug = self.create_publisher(
            Float64ArrayStamped, '/fdi/debug', 10
        )

        # ── Timer ───────────────────────────────────────────
        self.create_timer(self._dt, self._update)

        self.get_logger().info(
            f'FDINode started | rate={rate:.0f} Hz | '
            f'joint={self._joint_name} | '
            f'thresh_force={self.get_parameter("thresh_force").value} Nm | '
            f'thresh_encoder={self.get_parameter("thresh_encoder").value} rad | '
            f'debounce={self.get_parameter("debounce_count").value} samples'
        )

    # ─────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────

    def _js_cb(self, msg):
        try:
            idx = msg.name.index(self._joint_name)
        except ValueError:
            return
        self._theta_meas = msg.position[idx]
        self._theta_dot_meas = msg.velocity[idx] if idx < len(msg.velocity) else 0.0
        self._js_received = True

    def _tau_ext_cb(self, msg):
        self._tau_ext_meas = msg.data
        self._tau_ext_received = True

    def _tau_m_cb(self, msg):
        self._tau_m = msg.data
        self._tau_m_received = True

    def _ff_cb(self, msg):
        d = msg.data
        if len(d) > 0:
            self._M_eff = max(d[0], 1e-6)
        if len(d) > 1:
            self._proj = d[1]
        if len(d) > 3:
            self._tau_pass = d[3]
        self._ff_received = True

    def _traj_cb(self, msg):
        d = msg.data
        if len(d) < 3:
            return
        self._theta_ref = d[0]
        self._theta_dot_ref = d[1]
        self._theta_ddot_ref = d[2]
        self._traj_received = True

    # ─────────────────────────────────────────────────────────
    # Main update loop
    # ─────────────────────────────────────────────────────────

    def _update(self):
        if not (self._js_received and self._tau_ext_received and
                self._tau_m_received and self._ff_received and
                self._traj_received):
            return

        dt = self._dt

        # ── Read parameters ─────────────────────────────────
        fric_visc = self.get_parameter('fric_visc').value
        fric_coul = self.get_parameter('fric_coul').value
        fric_eps = self.get_parameter('fric_eps').value
        damping = self.get_parameter('damping_theta').value
        alpha = self.get_parameter('residual_alpha').value
        debounce_n = int(self.get_parameter('debounce_count').value)

        # ─────────────────────────────────────────────────────
        # R_FORCE — Admittance inversion (channel 0)
        # ─────────────────────────────────────────────────────
        # Reconstruct tau_ext from the trajectory reference.
        # Only valid when controller is not saturated and outside deadband.

        Mv = self.get_parameter('adm_M_virt').value
        Dv = self.get_parameter('adm_D_virt').value
        Kv = self.get_parameter('adm_K_virt').value
        theta_eq = self.get_parameter('adm_theta_eq').value
        deadband = self.get_parameter('adm_force_deadband').value
        th_min = self.get_parameter('adm_theta_ref_min').value
        th_max = self.get_parameter('adm_theta_ref_max').value
        w_buf = self.get_parameter('adm_wall_buffer').value
        K_w = self.get_parameter('adm_K_wall').value
        D_w = self.get_parameter('adm_D_wall').value

        # Saturation detection
        sat_margin = 1e-3
        ref_pos_saturated = (
            self._theta_ref <= th_min + sat_margin or
            self._theta_ref >= th_max - sat_margin
        )
        ref_vel_saturated = abs(self._theta_dot_ref) >= 5.0 - sat_margin
        in_deadband = abs(self._tau_ext_meas) < deadband

        self._force_valid = not (ref_pos_saturated or ref_vel_saturated or in_deadband)

        # Reconstruct tau_wall
        upper_threshold = th_max - w_buf
        lower_threshold = th_min + w_buf
        tau_wall_hat = 0.0
        if self._theta_ref > upper_threshold:
            penetration = self._theta_ref - upper_threshold
            vel_into_wall = max(self._theta_dot_ref, 0.0)
            tau_wall_hat = -K_w * penetration - D_w * vel_into_wall
        elif self._theta_ref < lower_threshold:
            penetration = lower_threshold - self._theta_ref
            vel_into_wall = max(-self._theta_dot_ref, 0.0)
            tau_wall_hat = K_w * penetration + D_w * vel_into_wall

        self._tau_wall_hat = tau_wall_hat

        if self._force_valid:
            self._tau_ext_hat = (
                Mv * self._theta_ddot_ref +
                Dv * self._theta_dot_ref +
                Kv * (self._theta_ref - theta_eq) -
                tau_wall_hat
            )
            self._r_force = self._tau_ext_meas - self._tau_ext_hat
        else:
            self._tau_ext_hat = 0.0
            self._r_force = 0.0

        # ─────────────────────────────────────────────────────
        # R_ENCODER — Luenberger observer (channel 3)
        # ─────────────────────────────────────────────────────
        # Predicts theta_hat from plant model driven by tau_m, tau_ext_meas.
        # Sensitive to encoder faults, immune to tau_ext faults.

        p1 = self.get_parameter('enc_obs_pole_1').value
        p2 = self.get_parameter('enc_obs_pole_2').value

        D_lin = fric_visc + damping
        D_over_M = D_lin / self._M_eff

        l1 = -(p1 + p2) - D_over_M
        l2 = p1 * p2 - l1 * D_over_M

        if not self._enc_obs_initialized:
            self._theta_hat = self._theta_meas
            self._theta_dot_hat = self._theta_dot_meas
            self._enc_obs_initialized = True

        # Friction on theta_dot_hat
        tau_fric_visc = fric_visc * self._theta_dot_hat
        tau_fric_coul = fric_coul * math.tanh(self._theta_dot_hat / fric_eps)
        tau_damp = damping * self._theta_dot_hat
        tau_dissipative = tau_fric_visc + tau_fric_coul + tau_damp

        theta_ddot_model = (
            self._tau_m
            + self._tau_pass
            + self._tau_ext_meas
            - self._proj
            - tau_dissipative
        ) / self._M_eff

        # Explicit Euler with Luenberger correction
        theta_dot_hat_new = (
            self._theta_dot_hat +
            dt * (theta_ddot_model + l2 * (self._theta_meas - self._theta_hat))
        )
        theta_hat_new = (
            self._theta_hat +
            dt * (self._theta_dot_hat + l1 * (self._theta_meas - self._theta_hat))
        )
        self._theta_dot_hat = theta_dot_hat_new
        self._theta_hat = theta_hat_new

        self._r_encoder = self._theta_meas - self._theta_hat

        # ─────────────────────────────────────────────────────
        # EMA filtering on absolute residuals
        # ─────────────────────────────────────────────────────
        if self._force_valid:
            self._r_force_filtered = ema(
                self._r_force_filtered, abs(self._r_force), alpha
            )
        else:
            # Decay filtered residual toward zero when invalid
            self._r_force_filtered = ema(self._r_force_filtered, 0.0, alpha)

        self._r_encoder_filtered = ema(
            self._r_encoder_filtered, abs(self._r_encoder), alpha
        )

        # ─────────────────────────────────────────────────────
        # Diagnosis with debounce + hold timer
        # ─────────────────────────────────────────────────────
        thresh_force = self.get_parameter('thresh_force').value
        thresh_encoder = self.get_parameter('thresh_encoder').value

        force_alert = self._r_force_filtered > thresh_force
        encoder_alert = self._r_encoder_filtered > thresh_encoder

        # Force debounce
        if force_alert:
            self._debounce_force = min(self._debounce_force + 1, debounce_n * 2)
        else:
            self._debounce_force = max(self._debounce_force - 1, 0)

        # Encoder debounce
        if encoder_alert:
            self._debounce_encoder = min(self._debounce_encoder + 1, debounce_n * 2)
        else:
            self._debounce_encoder = max(self._debounce_encoder - 1, 0)

        force_confirmed = self._debounce_force >= debounce_n
        encoder_confirmed = self._debounce_encoder >= debounce_n

        # Hold timer decrement
        if self._hold_steps > 0:
            self._hold_steps -= 1

        # Raw diagnosis from residuals
        if force_confirmed and not encoder_confirmed:
            raw_diag = Diagnosis.FAULT_CH0
        elif encoder_confirmed and not force_confirmed:
            raw_diag = Diagnosis.FAULT_CH3
        elif force_confirmed and encoder_confirmed:
            r_force_rel = self._r_force_filtered / max(thresh_force, 1e-9)
            r_enc_rel = self._r_encoder_filtered / max(thresh_encoder, 1e-9)
            raw_diag = (
                Diagnosis.FAULT_CH0
                if r_force_rel >= r_enc_rel
                else Diagnosis.FAULT_CH3
            )
        else:
            raw_diag = Diagnosis.HEALTHY

        # Hold: suppress cross-channel switches (Ch0 ↔ Ch3) during hold
        # to prevent oscillation. Always allow HEALTHY recovery.
        if (self._hold_steps > 0
                and raw_diag != Diagnosis.HEALTHY
                and self._diagnosis != Diagnosis.HEALTHY
                and raw_diag != self._diagnosis):
            new_diag = self._diagnosis
        else:
            new_diag = raw_diag

        # Start hold only on initial detection (HEALTHY → FAULT)
        if new_diag != Diagnosis.HEALTHY and self._diagnosis == Diagnosis.HEALTHY:
            hold_steps = int(self.get_parameter('hold_time_ms').value / (self._dt * 1000))
            self._hold_steps = hold_steps

        # Log transitions
        if new_diag != self._diagnosis:
            self.get_logger().info(
                f'FDI transition: {DIAG_NAMES[self._diagnosis]} -> '
                f'{DIAG_NAMES[new_diag]} | '
                f'r_force={self._r_force_filtered:.4f} '
                f'(thresh={thresh_force}) | '
                f'r_encoder={self._r_encoder_filtered:.6f} '
                f'(thresh={thresh_encoder})'
            )
            self._diagnosis = new_diag

        # ─────────────────────────────────────────────────────
        # Publish
        # ─────────────────────────────────────────────────────
        # Residuals
        residuals_msg = Float64MultiArray()
        residuals_msg.data = [self._r_force, self._r_encoder]
        self._pub_residuals.publish(residuals_msg)

        # Diagnosis string
        diag_msg = String()
        diag_msg.data = DIAG_NAMES[self._diagnosis]
        self._pub_diagnosis.publish(diag_msg)

        # Debug
        stamp = self.get_clock().now().to_msg()
        dbg = Float64ArrayStamped()
        dbg.header.stamp = stamp
        dbg.data = [
            self._theta_meas,           # [0]
            self._theta_hat,            # [1]
            self._theta_dot_meas,       # [2]
            self._theta_dot_hat,        # [3]
            self._tau_ext_meas,         # [4]
            self._tau_ext_hat,          # [5]
            self._r_force,              # [6]
            self._r_encoder,            # [7]
            self._r_force_filtered,     # [8]
            self._r_encoder_filtered,   # [9]
            1.0 if self._force_valid else 0.0,  # [10]
            float(self._diagnosis),     # [11]
        ]
        self._pub_debug.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = FDINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
