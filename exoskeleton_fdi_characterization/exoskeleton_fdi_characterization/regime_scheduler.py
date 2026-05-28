#!/usr/bin/env python3
"""
regime_scheduler.py — multi-regime baseline driver
====================================================

Cycles through a YAML-defined sequence of operating regimes during the
healthy-baseline recording. For each regime, sets the parameters of
the input node (external_wrench_pub by default) via the standard
/<node>/set_parameters service.

Three kinds of regime are supported:

  static    constant parameters held for 'duration' seconds.

  ramp      linear ramp from 'params' to 'end_params' discretized into
            'n_steps' equal-length sub-windows. Each sub-window is a
            small static segment.

  step_mix  cycles through a list of N parameter sets, each held for
            'step_duration' seconds, until 'duration' is reached.

For every parameter change the scheduler writes a START record in
the manifest (JSON list of dicts, one per active sub-window).

When the full sequence completes the scheduler logs 'BASELINE
COMPLETE' and stops emitting parameter updates. The recorded bag
should be closed manually (Ctrl+C on `ros2 bag record`) or the
launch file can stop the recorder when this node terminates.

ROS 2 PARAMETERS
----------------
  target_node      (str, default '/input')   target node for parameter sets
  param_frequency  (str, default 'frequency')
  param_f_min      (str, default 'f_min')
  param_f_max      (str, default 'f_max')
  settle_time      (float, default 0.5)      seconds after each set call
  manifest_path    (str)                     output JSON path
  sequence_json    (str)                     YAML-encoded sequence (see config)
"""
import json
import os
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


class RegimeScheduler(Node):

    def __init__(self):
        super().__init__('regime_scheduler')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('target_node',     '/input')
        self.declare_parameter('param_frequency', 'frequency')
        self.declare_parameter('param_f_min',     'f_min')
        self.declare_parameter('param_f_max',     'f_max')
        self.declare_parameter('settle_time',     0.5)
        self.declare_parameter('manifest_path',   'baseline_manifest.json')
        self.declare_parameter('sequence_json',   '[]')

        self.target_node    = str(self.get_parameter('target_node').value)
        self.param_freq     = str(self.get_parameter('param_frequency').value)
        self.param_fmin     = str(self.get_parameter('param_f_min').value)
        self.param_fmax     = str(self.get_parameter('param_f_max').value)
        self.settle_time    = float(self.get_parameter('settle_time').value)
        self.manifest_path  = str(self.get_parameter('manifest_path').value)
        sequence_str        = str(self.get_parameter('sequence_json').value)

        try:
            self.sequence = json.loads(sequence_str)
        except Exception as e:
            self.get_logger().error(f'Failed to parse sequence_json: {e}')
            self.sequence = []

        if not isinstance(self.sequence, list) or len(self.sequence) == 0:
            self.get_logger().error('Empty or invalid regime sequence; nothing to do.')
            return

        # ── Service client to the target node ───────────────────────
        # Standard ROS 2 service auto-advertised by every node.
        srv_name = self.target_node.rstrip('/') + '/set_parameters'
        self._set_params_cli = self.create_client(SetParameters, srv_name)

        self.get_logger().info(
            f'RegimeScheduler waiting for service: {srv_name}'
        )
        if not self._set_params_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f'Service {srv_name} unavailable after 10s. '
                f'Is the target node "{self.target_node}" running?'
            )
            return
        self.get_logger().info(f'RegimeScheduler connected to {self.target_node}')

        # ── Manifest writer ─────────────────────────────────────────
        self._manifest = []
        self._manifest_lock = threading.Lock()
        self._t0 = self.get_clock().now().nanoseconds * 1e-9
        self._running = True

        # ── Run the schedule in a background thread ─────────────────
        # We do not block the executor: the node continues to spin so
        # that the manifest can be flushed and ROS log messages still
        # get processed.
        self._thread = threading.Thread(target=self._run_schedule, daemon=True)
        self._thread.start()

    # ════════════════════════════════════════════════════════════════
    # Manifest helpers
    # ════════════════════════════════════════════════════════════════
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _record(self, regime_name: str, params: dict, kind: str, sub_index: int = 0):
        """Append a START marker for a (sub)regime to the manifest."""
        entry = {
            'name': regime_name,
            'kind': kind,
            'sub_index': sub_index,
            't_start': self._now(),
            't_start_relative': self._now() - self._t0,
            'frequency': float(params.get('frequency', 0.0)),
            'f_min':     float(params.get('f_min', 0.0)),
            'f_max':     float(params.get('f_max', 0.0)),
        }
        with self._manifest_lock:
            self._manifest.append(entry)
        self._flush_manifest()

    def _flush_manifest(self):
        try:
            path = Path(self.manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._manifest_lock:
                with open(path, 'w') as f:
                    json.dump(self._manifest, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'Failed to flush manifest: {e}')

    # ════════════════════════════════════════════════════════════════
    # Parameter set
    # ════════════════════════════════════════════════════════════════
    def _set_params(self, params: dict) -> bool:
        """
        Sends a SetParameters request with the three input parameters.
        Returns True iff every parameter was accepted.
        """
        req = SetParameters.Request()
        req.parameters = [
            self._make_double_param(self.param_freq, float(params.get('frequency', 0.05))),
            self._make_double_param(self.param_fmin, float(params.get('f_min', -12.0))),
            self._make_double_param(self.param_fmax, float(params.get('f_max', 0.0))),
        ]

        future = self._set_params_cli.call_async(req)

        # Block this thread until the future completes. The main rclpy
        # executor runs in a separate thread (the standard rclpy.spin in
        # main()), so this works because we can't deadlock on ourselves.
        timeout = 2.0
        t_start = time.monotonic()
        while not future.done():
            if (time.monotonic() - t_start) > timeout:
                self.get_logger().warn(
                    f'set_parameters timed out after {timeout}s for '
                    f'{self.target_node}'
                )
                return False
            time.sleep(0.01)

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'set_parameters failed: {e}')
            return False

        ok = all(r.successful for r in res.results)
        if not ok:
            for r in res.results:
                if not r.successful:
                    self.get_logger().warn(
                        f'parameter rejected: {r.reason}'
                    )
        return ok

    @staticmethod
    def _make_double_param(name: str, value: float) -> Parameter:
        p = Parameter()
        p.name = name
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_DOUBLE
        p.value.double_value = float(value)
        return p

    # ════════════════════════════════════════════════════════════════
    # Schedule execution
    # ════════════════════════════════════════════════════════════════
    def _run_schedule(self):
        """Iterate the YAML-defined sequence, blocking with sleep()."""
        self.get_logger().info(
            f'RegimeScheduler: starting schedule with '
            f'{len(self.sequence)} regimes'
        )
        for idx, regime in enumerate(self.sequence):
            if not self._running:
                break
            try:
                self._run_regime(idx, regime)
            except Exception as e:
                self.get_logger().error(
                    f'Regime {idx} ({regime.get("name", "?")}) '
                    f'crashed: {e}'
                )

        self.get_logger().info(
            'RegimeScheduler: BASELINE COMPLETE. '
            'You may stop the bag recorder now.'
        )
        self._running = False

    def _run_regime(self, idx: int, regime: dict):
        kind = str(regime.get('kind', 'static')).lower()
        name = str(regime.get('name', f'regime_{idx}'))
        duration = float(regime.get('duration', 60.0))

        self.get_logger().info(
            f'[{idx+1}/{len(self.sequence)}] {name} '
            f'({kind}, {duration:.0f}s)'
        )

        if kind == 'static':
            params = dict(regime.get('params', {}))
            self._set_params(params)
            self._record(name, params, kind)
            self._sleep(duration)

        elif kind == 'ramp':
            n = max(1, int(regime.get('n_steps', 16)))
            sub_dur = duration / n
            p0 = dict(regime.get('params', {}))
            p1 = dict(regime.get('end_params', p0))
            for k in range(n):
                if not self._running:
                    return
                alpha = (k + 0.5) / n  # midpoint of sub-step
                params = {
                    key: (1.0 - alpha) * p0.get(key, 0.0)
                            +     alpha  * p1.get(key, 0.0)
                    for key in ('frequency', 'f_min', 'f_max')
                }
                self._set_params(params)
                self._record(name, params, kind, sub_index=k)
                self._sleep(sub_dur)

        elif kind == 'step_mix':
            cycle = list(regime.get('cycle', []))
            if not cycle:
                self.get_logger().warn(f'{name}: empty cycle, skipping')
                return
            step_dur = float(regime.get('step_duration', 30.0))
            t_end = self._now() + duration
            k = 0
            while self._now() < t_end and self._running:
                params = dict(cycle[k % len(cycle)])
                self._set_params(params)
                self._record(name, params, kind, sub_index=k)
                remaining = t_end - self._now()
                self._sleep(min(step_dur, remaining))
                k += 1

        else:
            self.get_logger().warn(f'Unknown regime kind: {kind}')

    def _sleep(self, seconds: float):
        """Sleep in small slices so we can react quickly to shutdown."""
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end and self._running:
            time.sleep(min(0.5, max(0.0, end - time.monotonic())))


def main(args=None):
    rclpy.init(args=args)
    node = RegimeScheduler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node._flush_manifest()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()