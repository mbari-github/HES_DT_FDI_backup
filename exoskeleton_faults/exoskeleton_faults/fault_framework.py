#!/usr/bin/env python3
"""
fault_framework.py — Multi-channel fault injection framework
=============================================================

Single-node fault injector that proxies ALL FOUR fault channels
simultaneously and permanently. The 'active_channel' parameter selects
which channel is currently armed; the other three pass through transparently.

This design enables runtime channel switching via `ros2 param set`,
which is required for automated fault-injection sweeps. No node restart
or topic remapping change is ever needed during a test session.

PROXY TOPOLOGY
--------------
For every channel the framework subscribes to a "_raw" topic and republishes
to a canonical topic that all consumers read.

  Channel 0 — load cell / external force
    /exo_dynamics/tau_ext_theta_raw   →  /exo_dynamics/tau_ext_theta
  Channel 1 — admittance trajectory reference
    /trajectory_ref_raw                →  /trajectory_ref_post_inject
  Channel 2 — controller torque command
    /torque_raw                        →  /torque_post_inject
  Channel 3 — joint state encoder
    /joint_states_raw                  →  /joint_states

Channels 1 and 2 publish to *_post_inject so the bridge can subscribe to
them and continue applying its safety clamping downstream.

Channels 0 and 3 publish to the canonical sensor topic that all consumers
(controllers, observers, FDI) already read. With Opzione B naming the
producers (dynamics) must publish on the *_raw topic instead of the
canonical name.

ARMED vs PASSTHROUGH
--------------------
Only the channel matching `active_channel` is armed. Armed means the
fault pipeline is applied to the input value. The other three channels
forward the input unchanged at the same publication rate.

Setting `active_channel = -1` disarms all channels (pure passthrough).
This is the default at startup.

CHANNEL SWITCHING AT RUNTIME
----------------------------
  ros2 param set /fault_injector active_channel 3
  ros2 param set /fault_injector fault_type offset
  ros2 param set /fault_injector fault_magnitude 0.05
  ros2 param set /fault_injector fault_active true

When active_channel changes, all transient state (drift counters, delay
buffers, frozen values, spike timers) is reset, so the previous channel
returns cleanly to passthrough.

FAULT TYPES
-----------
Identical to the previous single-channel implementation:
  none, offset, scale, noise, freeze, spike, drift_linear, drift_parabolic,
  bias_drift, quantization, deadzone, saturation, dropout, delay.

PIPELINE CONFIGURATION
----------------------
Two configuration modes are supported:

  A) Legacy single-fault parameters (recommended for automated sweeps):
       fault_type, fault_magnitude, noise_std, spike_duration

  B) Pipeline JSON string (recommended for realistic compound faults):
       fault_pipeline_config: '[{"type":"bias_drift","std":0.001},
                                {"type":"noise","std":0.01}]'

If `fault_pipeline_config` is a non-empty list, mode B is used.
Otherwise the legacy parameters are used (mode A).

ROS 2 PARAMETERS
----------------
  active_channel        (int,    default -1)      currently armed channel (-1, 0, 1, 2, 3)
  fault_active          (bool,   default False)   global on/off for the armed channel
  fault_type            (str,    default 'offset')
  fault_magnitude       (float,  default 1.0)
  noise_std             (float,  default 0.1)
  spike_duration        (float,  default 0.05)
  fault_pipeline_config (str,    default '[]')    JSON pipeline (overrides legacy if non-empty)

  joint_name            (str,    default 'rev_crank')   Ch3 only: which joint is faulted
  fault_js_field        (str,    default 'position')    Ch3 only: 'position', 'velocity', 'both'
  target_index          (int,    default 0)             Ch1 only: which entry of the array

  publish_rate          (float,  default 200.0)         status publication rate

PUBLISHED TOPICS
----------------
  All four canonical channel topics (always, at the input rate of each producer)
  /fault_injector/status   exoskeleton_safety_msgs/Float64ArrayStamped (200 Hz)
"""

import json
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from exoskeleton_safety_msgs.msg import Float64Stamped, Float64ArrayStamped


# ════════════════════════════════════════════════════════════════════
# CHANNEL TABLE
# ════════════════════════════════════════════════════════════════════
# Each entry describes a single proxy hop:
#   sub_topic : where the framework reads from (the *_raw side)
#   pub_topic : where the framework publishes to (the canonical side)
#   msg_type  : ROS 2 message type
#   description : human-readable label for logs
#
# The order is fixed: index 0..3 must match the channel number.
CHANNELS = {
    0: {
        'sub_topic': '/exo_dynamics/tau_ext_theta_raw',
        'pub_topic': '/exo_dynamics/tau_ext_theta',
        'msg_type':  'Float64Stamped',
        'description': 'External force projected on theta (load cell)',
    },
    1: {
        'sub_topic': '/trajectory_ref_raw',
        'pub_topic': '/trajectory_ref_post_inject',
        'msg_type':  'Float64ArrayStamped',
        'description': 'Trajectory reference [theta_ref, theta_dot_ref, theta_ddot_ref]',
    },
    2: {
        'sub_topic': '/torque_raw',
        'pub_topic': '/torque_post_inject',
        'msg_type':  'Float64Stamped',
        'description': 'Commanded motor torque (inner loop output)',
    },
    3: {
        'sub_topic': '/joint_states_raw',
        'pub_topic': '/joint_states',
        'msg_type':  'JointState',
        'description': 'Joint position/velocity feedback (encoder)',
    },
}

FAULT_TYPE_MAP = {
    'none':            0,
    'offset':          1,
    'scale':           2,
    'noise':           3,
    'freeze':          4,
    'spike':           5,
    'drift_linear':    6,
    'drift_parabolic': 7,
    'bias_drift':      8,
    'quantization':    9,
    'deadzone':       10,
    'saturation':     11,
    'dropout':        12,
    'delay':          13,
}


# ════════════════════════════════════════════════════════════════════
# PER-CHANNEL TRANSIENT STATE
# ════════════════════════════════════════════════════════════════════
# Each channel keeps an isolated copy of all stateful fault types so that
# switching active_channel cleanly resets the previously armed channel.
def make_channel_state():
    return {
        'last_raw':       0.0,
        'last_faulted':   0.0,
        'n_injections':   0,

        # transient states for each stateful fault type
        'bias_drift':     {'bias':       0.0},
        'drift':          {'step':       0},
        'delay':          {'buffer':     deque(maxlen=10000)},
        'spike':          {'active':     False, 'start_time': None},
        'frozen_value':   None,
        'freeze_logged':  False,
    }


class FaultFramework(Node):
    """Multi-channel fault injector with runtime channel switching."""

    # ════════════════════════════════════════════════════════════════
    # CONSTRUCTION
    # ════════════════════════════════════════════════════════════════
    def __init__(self):
        super().__init__('fault_injector')

        # ── Parameters ──────────────────────────────────────────────
        # Global control
        self.declare_parameter('active_channel',         -1)
        self.declare_parameter('fault_active',           False)

        # Legacy single-fault parameters
        self.declare_parameter('fault_type',             'offset')
        self.declare_parameter('fault_magnitude',        1.0)
        self.declare_parameter('noise_std',              0.1)
        self.declare_parameter('spike_duration',         0.05)

        # Pipeline override (JSON list)
        self.declare_parameter('fault_pipeline_config',  '[]')

        # Channel-specific
        self.declare_parameter('joint_name',             'rev_crank')
        self.declare_parameter('fault_js_field',         'position')
        self.declare_parameter('target_index',           0)

        # Status publication rate
        self.declare_parameter('publish_rate',           200.0)

        # ── Per-channel transient state ─────────────────────────────
        self._state = {ch: make_channel_state() for ch in CHANNELS}

        # ── Track the currently armed channel for change detection ──
        self._last_active_channel = self._read_active_channel()
        self._last_pipeline_str = ''

        # ── ROS I/O — instantiate ALL FOUR proxies ──────────────────
        self._subs = {}
        self._pubs = {}
        for ch, info in CHANNELS.items():
            self._make_proxy(ch, info)

        # ── Status publisher (single, channel-agnostic) ─────────────
        self._pub_status = self.create_publisher(
            Float64ArrayStamped, '/fault_injector/status', 10
        )
        publish_rate = float(self.get_parameter('publish_rate').value)
        self._timer_status = self.create_timer(
            1.0 / publish_rate, self._publish_status
        )

        self.get_logger().info(
            f'FaultFramework started — multi-channel proxy ACTIVE on all 4 channels.\n'
            f'  active_channel : {self._last_active_channel} '
            f'(-1 = passthrough only)\n'
            f'  fault_active   : {self.get_parameter("fault_active").value}\n'
            f'\nRuntime control:\n'
            f'  ros2 param set /fault_injector active_channel <0|1|2|3|-1>\n'
            f'  ros2 param set /fault_injector fault_type <type>\n'
            f'  ros2 param set /fault_injector fault_magnitude <m>\n'
            f'  ros2 param set /fault_injector fault_active true'
        )

    def _make_proxy(self, channel: int, info: dict):
        """Instantiate the subscriber+publisher pair for one channel."""
        msg_type = info['msg_type']

        if msg_type == 'Float64Stamped':
            self._subs[channel] = self.create_subscription(
                Float64Stamped, info['sub_topic'],
                lambda msg, ch=channel: self._cb_float64(msg, ch),
                10,
            )
            self._pubs[channel] = self.create_publisher(
                Float64Stamped, info['pub_topic'], 10
            )

        elif msg_type == 'Float64ArrayStamped':
            self._subs[channel] = self.create_subscription(
                Float64ArrayStamped, info['sub_topic'],
                lambda msg, ch=channel: self._cb_multiarray(msg, ch),
                10,
            )
            self._pubs[channel] = self.create_publisher(
                Float64ArrayStamped, info['pub_topic'], 10
            )

        elif msg_type == 'JointState':
            self._subs[channel] = self.create_subscription(
                JointState, info['sub_topic'],
                lambda msg, ch=channel: self._cb_joint_states(msg, ch),
                10,
            )
            self._pubs[channel] = self.create_publisher(
                JointState, info['pub_topic'], 10
            )

        else:
            raise RuntimeError(f'Unknown msg_type {msg_type} for channel {channel}')

    # ════════════════════════════════════════════════════════════════
    # PARAMETER ACCESS WITH CHANGE DETECTION
    # ════════════════════════════════════════════════════════════════
    def _read_active_channel(self) -> int:
        ch = int(self.get_parameter('active_channel').value)
        if ch != -1 and ch not in CHANNELS:
            self.get_logger().warn(
                f'Invalid active_channel={ch}, treating as -1 (no channel armed).'
            )
            return -1
        return ch

    def _check_active_channel_change(self):
        """If active_channel changed since last check, reset state on the
        previously-armed channel. This guarantees that swapping channels
        leaves no leftover transient state behind."""
        current = self._read_active_channel()
        if current != self._last_active_channel:
            prev = self._last_active_channel
            if prev != -1:
                self._reset_channel_state(prev)
            self.get_logger().info(
                f'FaultFramework: active_channel changed {prev} -> {current}. '
                f'Channel {prev} reset to passthrough.'
            )
            self._last_active_channel = current

    def _check_pipeline_change(self):
        """Reset transient states when the pipeline JSON changes."""
        cfg_str = str(self.get_parameter('fault_pipeline_config').value)
        if cfg_str != self._last_pipeline_str:
            ch = self._last_active_channel
            if ch != -1:
                self._reset_channel_state(ch)
            self._last_pipeline_str = cfg_str

    def _reset_channel_state(self, channel: int):
        """Reset transient state for one channel (preserves last_raw/last_faulted)."""
        s = self._state[channel]
        s['bias_drift']['bias'] = 0.0
        s['drift']['step'] = 0
        s['delay']['buffer'].clear()
        s['spike']['active'] = False
        s['spike']['start_time'] = None
        s['frozen_value'] = None
        s['freeze_logged'] = False

    # ════════════════════════════════════════════════════════════════
    # PIPELINE PARSING
    # ════════════════════════════════════════════════════════════════
    def _get_pipeline(self) -> list:
        """
        Returns the active fault pipeline as a list of stage dicts.
        Pipeline JSON takes precedence; otherwise falls back to legacy
        single-fault parameters.
        """
        cfg_str = str(self.get_parameter('fault_pipeline_config').value)
        try:
            pipeline = json.loads(cfg_str)
            if isinstance(pipeline, list) and len(pipeline) > 0:
                return pipeline
        except Exception:
            pass

        # Legacy single-fault fallback
        return [{
            'type':       str(self.get_parameter('fault_type').value),
            'magnitude':  float(self.get_parameter('fault_magnitude').value),
            'noise_std':  float(self.get_parameter('noise_std').value),
            'duration':   float(self.get_parameter('spike_duration').value),
        }]

    def _is_armed(self, channel: int) -> bool:
        """A channel is armed when it matches active_channel and fault_active is True."""
        if channel != self._last_active_channel:
            return False
        return bool(self.get_parameter('fault_active').value)

    def _is_hard_freeze_active(self, pipeline: list) -> bool:
        """Returns True iff any pipeline stage is a 'freeze'."""
        return any(stage.get('type', '') == 'freeze' for stage in pipeline)

    # ════════════════════════════════════════════════════════════════
    # FAULT APPLICATION
    # ════════════════════════════════════════════════════════════════
    def _apply_pipeline(self, value: float, pipeline: list, channel: int) -> float:
        """Apply each fault stage sequentially. Output of stage N is input of N+1."""
        out = value
        for stage in pipeline:
            out = self._apply_single_fault(stage, out, channel)
        return out

    def _apply_single_fault(self, cfg: dict, value: float, channel: int) -> float:
        """Apply one fault stage to a scalar. State is per-channel."""
        s = self._state[channel]
        ftype = cfg.get('type', 'none')

        if ftype == 'none':
            return value

        # ── Simple additive / multiplicative ────────────────────────
        if ftype == 'offset':
            s['n_injections'] += 1
            # 'magnitude' is the legacy parameter name, fallback for compatibility
            return value + float(cfg.get('magnitude', cfg.get('offset', 0.0)))

        if ftype == 'scale':
            s['n_injections'] += 1
            # 'factor' for pipeline; 'magnitude' for legacy
            return value * float(cfg.get('factor', cfg.get('magnitude', 1.0)))

        if ftype == 'noise':
            s['n_injections'] += 1
            std = float(cfg.get('std', cfg.get('noise_std', 0.1)))
            return value + float(np.random.normal(0.0, std))

        # ── Drift ────────────────────────────────────────────────────
        if ftype == 'drift_linear':
            s['drift']['step'] += 1
            s['n_injections'] += 1
            slope = float(cfg.get('slope', cfg.get('magnitude', 0.0)))
            return value + slope * s['drift']['step']

        if ftype == 'drift_parabolic':
            s['drift']['step'] += 1
            s['n_injections'] += 1
            coeff = float(cfg.get('coeff', cfg.get('magnitude', 0.0)))
            return value + coeff * (s['drift']['step'] ** 2)

        if ftype == 'bias_drift':
            std = float(cfg.get('std', 0.001))
            s['bias_drift']['bias'] += float(np.random.normal(0.0, std))
            s['n_injections'] += 1
            return value + s['bias_drift']['bias']

        # ── Sensor degradation ───────────────────────────────────────
        if ftype == 'quantization':
            res = float(cfg.get('resolution', 0.01))
            if res <= 0.0:
                return value
            s['n_injections'] += 1
            return res * round(value / res)

        if ftype == 'deadzone':
            eps = float(cfg.get('threshold', 0.01))
            s['n_injections'] += 1
            return 0.0 if abs(value) < eps else value

        if ftype == 'saturation':
            lim = float(cfg.get('limit', 1.0))
            s['n_injections'] += 1
            return float(np.clip(value, -lim, lim))

        if ftype == 'dropout':
            if float(np.random.rand()) < float(cfg.get('prob', 0.1)):
                s['n_injections'] += 1
                return s['last_faulted']
            return value

        if ftype == 'delay':
            n = int(cfg.get('samples', 5))
            s['delay']['buffer'].append(value)
            if len(s['delay']['buffer']) > n:
                s['n_injections'] += 1
                return s['delay']['buffer'].popleft()
            return value

        # ── Spike (transient) ────────────────────────────────────────
        if ftype == 'spike':
            duration = float(cfg.get('duration', 0.05))
            magnitude = float(cfg.get('magnitude', 1.0))
            now = self.get_clock().now().nanoseconds * 1e-9
            sp = s['spike']
            if not sp['active']:
                sp['active'] = True
                sp['start_time'] = now
            elapsed = now - sp['start_time']
            if elapsed < duration:
                s['n_injections'] += 1
                return value + magnitude
            else:
                sp['active'] = False
                sp['start_time'] = None
                return value

        # ── Freeze (handled at callback level, no-op here) ───────────
        if ftype == 'freeze':
            return value

        self.get_logger().warn(
            f'FaultFramework: unknown fault type "{ftype}" — skipping stage.'
        )
        return value

    # ════════════════════════════════════════════════════════════════
    # CALLBACKS — one per message type, dispatched by channel
    # ════════════════════════════════════════════════════════════════
    def _cb_float64(self, msg: Float64Stamped, channel: int):
        # Detect runtime parameter changes once per callback
        self._check_active_channel_change()
        self._check_pipeline_change()

        s = self._state[channel]
        raw = float(msg.data)
        s['last_raw'] = raw

        # ── Passthrough path ────────────────────────────────────────
        if not self._is_armed(channel):
            out = Float64Stamped()
            out.header = msg.header
            out.data = raw
            s['last_faulted'] = raw
            self._pubs[channel].publish(out)
            return

        # ── Armed: apply pipeline ───────────────────────────────────
        pipeline = self._get_pipeline()

        # Hard freeze: suppress publication entirely
        if self._is_hard_freeze_active(pipeline):
            if s['frozen_value'] is None:
                s['frozen_value'] = raw
                s['last_faulted'] = raw
            if not s['freeze_logged']:
                self.get_logger().info(
                    f'FaultFramework [FREEZE] blocking '
                    f'{CHANNELS[channel]["pub_topic"]} | last_raw={raw:.6f}'
                )
                s['freeze_logged'] = True
            s['n_injections'] += 1
            return

        faulted = self._apply_pipeline(raw, pipeline, channel)
        s['last_faulted'] = faulted

        out = Float64Stamped()
        out.header = msg.header
        out.data = faulted
        self._pubs[channel].publish(out)

    def _cb_multiarray(self, msg: Float64ArrayStamped, channel: int):
        self._check_active_channel_change()
        self._check_pipeline_change()

        s = self._state[channel]
        data = list(msg.data)
        if not data:
            if not self._is_armed(channel):
                self._pubs[channel].publish(msg)
            return

        target_idx = int(self.get_parameter('target_index').value)
        target_idx = max(0, min(target_idx, len(data) - 1))

        raw = float(data[target_idx])
        s['last_raw'] = raw

        # ── Passthrough path ────────────────────────────────────────
        if not self._is_armed(channel):
            out = Float64ArrayStamped()
            out.header = msg.header
            out.data = data
            s['last_faulted'] = raw
            self._pubs[channel].publish(out)
            return

        # ── Armed: apply pipeline ───────────────────────────────────
        pipeline = self._get_pipeline()

        if self._is_hard_freeze_active(pipeline):
            if s['frozen_value'] is None:
                s['frozen_value'] = raw
                s['last_faulted'] = raw
            if not s['freeze_logged']:
                self.get_logger().info(
                    f'FaultFramework [FREEZE] blocking '
                    f'{CHANNELS[channel]["pub_topic"]}'
                )
                s['freeze_logged'] = True
            s['n_injections'] += 1
            return

        faulted = self._apply_pipeline(raw, pipeline, channel)
        s['last_faulted'] = faulted
        data[target_idx] = faulted

        out = Float64ArrayStamped()
        out.header = msg.header
        out.data = data
        self._pubs[channel].publish(out)

    def _cb_joint_states(self, msg: JointState, channel: int):
        self._check_active_channel_change()
        self._check_pipeline_change()

        s = self._state[channel]
        joint_name = str(self.get_parameter('joint_name').value)
        js_field = str(self.get_parameter('fault_js_field').value)

        # Find the index of the target joint
        try:
            idx = list(msg.name).index(joint_name)
        except ValueError:
            # Target joint not found: passthrough untouched
            self._pubs[channel].publish(msg)
            return

        # ── Passthrough path ────────────────────────────────────────
        if not self._is_armed(channel):
            if idx < len(msg.position):
                s['last_raw'] = float(msg.position[idx])
                s['last_faulted'] = s['last_raw']
            self._pubs[channel].publish(msg)
            return

        # ── Armed: apply pipeline (with freeze handling) ────────────
        pipeline = self._get_pipeline()

        if self._is_hard_freeze_active(pipeline):
            if not s['freeze_logged']:
                self.get_logger().info(
                    f'FaultFramework [FREEZE] blocking '
                    f'{CHANNELS[channel]["pub_topic"]} | joint={joint_name}'
                )
                s['freeze_logged'] = True
            s['n_injections'] += 1
            return

        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)

        if js_field in ('position', 'both') and idx < len(out.position):
            raw_pos = float(out.position[idx])
            s['last_raw'] = raw_pos
            faulted_pos = self._apply_pipeline(raw_pos, pipeline, channel)
            s['last_faulted'] = faulted_pos
            out.position[idx] = faulted_pos

        if js_field in ('velocity', 'both') and idx < len(out.velocity):
            raw_vel = float(out.velocity[idx])
            if js_field == 'velocity':
                s['last_raw'] = raw_vel
                faulted_vel = self._apply_pipeline(raw_vel, pipeline, channel)
                s['last_faulted'] = faulted_vel
            else:
                # 'both': we already updated last_raw/last_faulted with position
                faulted_vel = self._apply_pipeline(raw_vel, pipeline, channel)
            out.velocity[idx] = faulted_vel

        self._pubs[channel].publish(out)

    # ════════════════════════════════════════════════════════════════
    # STATUS PUBLISHING
    # ════════════════════════════════════════════════════════════════
    def _publish_status(self):
        """
        Publishes diagnostic status on /fault_injector/status.

        Layout (Float64ArrayStamped):
          [0]  active_channel  (-1 if none)
          [1]  fault_type_id   (id of the FIRST stage of the active pipeline)
          [2]  fault_active    (1.0 / 0.0)
          [3]  fault_magnitude (first stage magnitude)
          [4]  n_injections    (cumulative count on the armed channel)
          [5]  last_raw        (raw value of the armed channel; 0 if disarmed)
          [6]  last_faulted    (faulted value of the armed channel)
          [7]  delta = last_faulted - last_raw
        """
        active_ch = self._last_active_channel
        fault_active = bool(self.get_parameter('fault_active').value)
        pipeline = self._get_pipeline()

        first_type = pipeline[0].get('type', 'none') if pipeline else 'none'
        fault_type_id = float(FAULT_TYPE_MAP.get(first_type, -1))
        first_magnitude = float(pipeline[0].get('magnitude', 0.0)) if pipeline else 0.0

        if active_ch == -1:
            n_inj = 0
            last_raw = 0.0
            last_faulted = 0.0
        else:
            s = self._state[active_ch]
            n_inj = s['n_injections']
            last_raw = s['last_raw']
            last_faulted = s['last_faulted']

        delta = last_faulted - last_raw

        msg = Float64ArrayStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = [
            float(active_ch),
            fault_type_id,
            1.0 if fault_active else 0.0,
            first_magnitude,
            float(n_inj),
            float(last_raw),
            float(last_faulted),
            float(delta),
        ]
        self._pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaultFramework()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()