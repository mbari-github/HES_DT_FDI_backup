#!/usr/bin/env python3
"""
fault_orchestrator.py — automated fault-injection sweep
========================================================

Drives the fault_injector through a YAML-defined sweep of
(channel x fault_type x magnitude) windows. For each window the
orchestrator:

  1. records a healthy_pre interval  (no fault armed)
  2. arms the fault                  (active_channel + type + magnitude)
  3. waits fault_duration            (fault is producing corruption)
  4. disarms the fault               (fault_active=false, channel=-1)
  5. records a healthy_post interval

Window metadata is appended to a JSON manifest with timestamps that
allow offline tools to slice the bag into per-fault sub-bags.

For special fault types ('noise' uses noise_std, 'spike' uses
spike_duration), the orchestrator also sets the corresponding
auxiliary parameter. The 'param' field in the sweep entry overrides
the default mapping.

ROS 2 PARAMETERS
----------------
  target_node              (str, default '/fault_injector')
  param_active_channel     (str, default 'active_channel')
  param_fault_active       (str, default 'fault_active')
  param_fault_type         (str, default 'fault_type')
  param_fault_magnitude    (str, default 'fault_magnitude')
  param_noise_std          (str, default 'noise_std')
  param_spike_duration     (str, default 'spike_duration')
  healthy_pre_duration     (float, default 10.0)
  fault_duration           (float, default 30.0)
  healthy_post_duration    (float, default 10.0)
  manifest_path            (str)
  sweep_json               (str)
"""
import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


class FaultOrchestrator(Node):

    def __init__(self):
        super().__init__('fault_orchestrator')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('target_node',          '/fault_injector')
        self.declare_parameter('param_active_channel', 'active_channel')
        self.declare_parameter('param_fault_active',   'fault_active')
        self.declare_parameter('param_fault_type',     'fault_type')
        self.declare_parameter('param_fault_magnitude','fault_magnitude')
        self.declare_parameter('param_noise_std',      'noise_std')
        self.declare_parameter('param_spike_duration', 'spike_duration')
        self.declare_parameter('healthy_pre_duration',  10.0)
        self.declare_parameter('fault_duration',        30.0)
        self.declare_parameter('healthy_post_duration', 10.0)
        self.declare_parameter('manifest_path',        'fault_sweep_manifest.json')
        self.declare_parameter('sweep_json',           '[]')

        self.target_node      = str(self.get_parameter('target_node').value)
        self.p_channel        = str(self.get_parameter('param_active_channel').value)
        self.p_active         = str(self.get_parameter('param_fault_active').value)
        self.p_type           = str(self.get_parameter('param_fault_type').value)
        self.p_magnitude      = str(self.get_parameter('param_fault_magnitude').value)
        self.p_noise          = str(self.get_parameter('param_noise_std').value)
        self.p_spike_dur      = str(self.get_parameter('param_spike_duration').value)
        self.healthy_pre      = float(self.get_parameter('healthy_pre_duration').value)
        self.fault_dur        = float(self.get_parameter('fault_duration').value)
        self.healthy_post     = float(self.get_parameter('healthy_post_duration').value)
        self.manifest_path    = str(self.get_parameter('manifest_path').value)

        sweep_str = str(self.get_parameter('sweep_json').value)
        try:
            self.sweep_spec = json.loads(sweep_str)
        except Exception as e:
            self.get_logger().error(f'Failed to parse sweep_json: {e}')
            self.sweep_spec = []

        # ── Build flat list of windows ──────────────────────────────
        # One window per (channel, type, magnitude). The 'param' field
        # in the sweep entry, if present, names the auxiliary parameter
        # to set (e.g. 'noise_std' for type='noise').
        self.windows = []
        for entry in self.sweep_spec:
            ch    = int(entry.get('channel', -1))
            ftype = str(entry.get('type', 'none'))
            param_name = entry.get('param', None)  # may be None
            for mag in entry.get('magnitudes', [0.0]):
                self.windows.append({
                    'channel':    ch,
                    'type':       ftype,
                    'magnitude':  float(mag),
                    'aux_param':  param_name,
                    'aux_value':  float(mag) if param_name else None,
                })

        if not self.windows:
            self.get_logger().error('Empty fault sweep; nothing to do.')
            return

        self.get_logger().info(
            f'FaultOrchestrator: {len(self.windows)} windows. '
            f'Estimated total: '
            f'{len(self.windows) * (self.healthy_pre + self.fault_dur + self.healthy_post) / 60.0:.1f} min'
        )

        # ── Service client ──────────────────────────────────────────
        srv_name = self.target_node.rstrip('/') + '/set_parameters'
        self._cli = self.create_client(SetParameters, srv_name)

        if not self._cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f'Service {srv_name} unavailable. '
                f'Is the fault_injector running?'
            )
            return

        # ── Manifest ────────────────────────────────────────────────
        self._manifest = []
        self._manifest_lock = threading.Lock()
        self._t0 = self._now()
        self._running = True

        # ── Run sweep in a background thread ────────────────────────
        self._thread = threading.Thread(target=self._run_sweep, daemon=True)
        self._thread.start()

    # ════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _flush_manifest(self):
        try:
            path = Path(self.manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._manifest_lock:
                with open(path, 'w') as f:
                    json.dump(self._manifest, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'Failed to flush manifest: {e}')

    def _make_int(self, name: str, v: int) -> Parameter:
        p = Parameter(); p.name = name
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_INTEGER
        p.value.integer_value = int(v)
        return p

    def _make_double(self, name: str, v: float) -> Parameter:
        p = Parameter(); p.name = name
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_DOUBLE
        p.value.double_value = float(v)
        return p

    def _make_bool(self, name: str, v: bool) -> Parameter:
        p = Parameter(); p.name = name
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_BOOL
        p.value.bool_value = bool(v)
        return p

    def _make_string(self, name: str, v: str) -> Parameter:
        p = Parameter(); p.name = name
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_STRING
        p.value.string_value = str(v)
        return p

    def _set_params(self, params: list, label: str = '') -> bool:
        req = SetParameters.Request()
        req.parameters = params
        future = self._cli.call_async(req)

        timeout = 2.0
        t_start = time.monotonic()
        while not future.done():
            if (time.monotonic() - t_start) > timeout:
                self.get_logger().warn(f'set_parameters timeout: {label}')
                return False
            time.sleep(0.01)

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'set_parameters failed ({label}): {e}')
            return False

        ok = all(r.successful for r in res.results)
        if not ok:
            for r in res.results:
                if not r.successful:
                    self.get_logger().warn(
                        f'param rejected ({label}): {r.reason}'
                    )
        return ok

    def _sleep(self, seconds: float):
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end and self._running:
            time.sleep(min(0.5, max(0.0, end - time.monotonic())))

    # ════════════════════════════════════════════════════════════════
    # Sweep execution
    # ════════════════════════════════════════════════════════════════
    def _disarm(self):
        """Set fault_active=False and active_channel=-1."""
        self._set_params([
            self._make_bool(self.p_active, False),
            self._make_int(self.p_channel, -1),
        ], label='disarm')

    def _arm(self, w: dict):
        """Configure and arm one fault window."""
        params = [
            self._make_int(self.p_channel,    int(w['channel'])),
            self._make_string(self.p_type,    str(w['type'])),
            self._make_double(self.p_magnitude, float(w['magnitude'])),
        ]
        if w['aux_param'] == 'noise_std':
            params.append(self._make_double(self.p_noise, float(w['aux_value'])))
        elif w['aux_param'] == 'spike_duration':
            params.append(self._make_double(self.p_spike_dur, float(w['aux_value'])))

        self._set_params(params, label=f'configure {w["type"]}@Ch{w["channel"]}')
        # Arm last, after all other params are set
        self._set_params(
            [self._make_bool(self.p_active, True)],
            label='arm',
        )

    def _name_window(self, w: dict) -> str:
        return f'Ch{w["channel"]}_{w["type"]}_{w["magnitude"]:g}'

    def _run_sweep(self):
        # Make sure we start clean
        self._disarm()
        self._sleep(2.0)

        for idx, w in enumerate(self.windows):
            if not self._running:
                break

            name = self._name_window(w)
            self.get_logger().info(
                f'[{idx+1}/{len(self.windows)}] {name}'
            )

            t_pre_start = self._now()
            self._sleep(self.healthy_pre)

            t_arm = self._now()
            self._arm(w)

            self._sleep(self.fault_dur)

            t_disarm = self._now()
            self._disarm()

            self._sleep(self.healthy_post)
            t_post_end = self._now()

            entry = {
                'name':          name,
                'index':         idx,
                'channel':       w['channel'],
                'type':          w['type'],
                'magnitude':     w['magnitude'],
                'aux_param':     w['aux_param'],
                'aux_value':     w['aux_value'],
                't_pre_start':   t_pre_start,
                't_arm':         t_arm,
                't_disarm':      t_disarm,
                't_post_end':    t_post_end,
                't_pre_start_relative':  t_pre_start - self._t0,
                't_arm_relative':        t_arm       - self._t0,
                't_disarm_relative':     t_disarm    - self._t0,
                't_post_end_relative':   t_post_end  - self._t0,
            }
            with self._manifest_lock:
                self._manifest.append(entry)
            self._flush_manifest()

        self.get_logger().info(
            'FaultOrchestrator: SWEEP COMPLETE. '
            'You may stop the bag recorder now.'
        )
        self._running = False


def main(args=None):
    rclpy.init(args=args)
    node = FaultOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        # Try to disarm on shutdown so the fault is not left active.
        try:
            node._disarm()
        except Exception:
            pass
        node._flush_manifest()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()