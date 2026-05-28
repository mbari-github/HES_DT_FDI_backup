"""
healthy_baseline.launch.py
===========================

Records the multi-regime healthy baseline with the fault framework
in permanent transparent passthrough (active_channel = -1, no fault
injected).

Why the fault framework is included here
-----------------------------------------
The production topology always has the framework in the signal path
(dynamics → *_raw → framework → canonical → consumers).  If we
calibrate thresholds from a baseline without the framework the proxy's
overhead (extra message copy, tiny timestamp shift) is absent from the
healthy residual distribution but present during sweep and production.
This causes a systematic underestimation of the healthy noise floor and
leads to thresholds that are too tight.

Including the framework in passthrough mode ensures that the
characterization topology is byte-for-byte identical to the production
and sweep topologies.  Any proxy artefacts are absorbed into the healthy
distribution and are therefore reflected in the derived thresholds.

Topology
--------
    dynamics  ──[*_raw]──► fault_framework ──[canonical]──► consumers
    admittance_controller  ──[trajectory_ref_raw]──► framework ──►
    trajectory_controller  ──[torque_raw]──► framework ──[torque_post_inject]──►
    fdi_node   reads post-inject and canonical topics (same as sweep)
    regime_scheduler  ──► /input/set_parameters  (drives input sinusoid)
    rosbag2 record  ──► records raw + post_inject + FDI topics

Output
------
    ./bags/healthy_baseline_<timestamp>/   rosbag2 archive
    ./baseline_manifest.json               regime transition log

Usage
-----
    ros2 launch exoskeleton_fdi_characterization healthy_baseline.launch.py

Press Ctrl-C when the scheduler logs 'BASELINE COMPLETE'.
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():

    description_pkg = get_package_share_directory('exoskeleton_description')
    bringup_pkg     = get_package_share_directory('exoskeleton_bringup')
    char_pkg        = get_package_share_directory('exoskeleton_fdi_characterization')
    fdi_pkg         = get_package_share_directory('exoskeleton_fdi')

    urdf_file = os.path.join(description_pkg, 'urdf', 'assembly_with_hand.urdf')
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    # ── Config files ──────────────────────────────────────────────────
    # REGIMES_CONFIG env var overrides the default for testing:
    #   REGIMES_CONFIG=/tmp/quick.yaml ros2 launch ... healthy_baseline.launch.py
    _default_regimes = os.path.join(char_pkg, 'config', 'healthy_regimes.yaml')
    regimes_yaml = os.environ.get('REGIMES_CONFIG', _default_regimes)
    fault_fw_params = os.path.join(bringup_pkg, 'config', 'fault_framework_params.yaml')
    fdi_params      = os.path.join(fdi_pkg,     'config', 'fdi_params.yaml')

    # ── Bag output ────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path  = os.path.join('bags', f'healthy_baseline_{timestamp}')

    # ── Topics to record ──────────────────────────────────────────────
    record_topics = [
        # Plant — raw (pre-framework) and canonical (post-framework)
        '/joint_states',
        '/joint_states_raw',
        '/exo_dynamics/tau_ext_theta',
        '/exo_dynamics/tau_ext_theta_raw',
        '/exo_dynamics/debug',
        '/exo_dynamics/ff_terms',
        '/exo_dynamics/external_wrench',
        # Control — raw, post-inject
        '/torque_raw',
        '/torque_post_inject',
        '/trajectory_ref_raw',
        '/trajectory_ref_post_inject',
        '/admittance/debug',
        '/traj_ctrl/debug',
        # FDI — residuals and diagnosis
        '/fdi/debug',
        '/fdi/diagnosis',
        '/fdi/residuals',
        # Framework ground truth (active_channel = -1, fault_active = false throughout)
        '/fault_injector/status',
    ]

    bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', bag_path] + record_topics,
        output='screen',
    )

    # ════════════════════════════════════════════════════════════════
    # Remaps — same as fault_sweep.launch.py so topology is identical
    # ════════════════════════════════════════════════════════════════

    # RSP reads raw joints so the TF tree stays correct.
    rsp_remaps = [('/joint_states', '/joint_states_raw')]

    # Dynamics publishes on *_raw so the framework can intercept.
    # Dynamics consumes the post-inject torque command from the framework.
    dynamics_remaps = [
        ('/joint_states',               '/joint_states_raw'),
        ('/exo_dynamics/tau_ext_theta', '/exo_dynamics/tau_ext_theta_raw'),
        ('/torque',                     '/torque_post_inject'),
    ]

    # Admittance publishes on _raw; reads canonical sensor topics.
    admittance_remaps = [('/trajectory_ref', '/trajectory_ref_raw')]

    # Trajectory reads the framework's reference output; publishes on _raw.
    trajectory_remaps = [
        ('/torque',         '/torque_raw'),
        ('/trajectory_ref', '/trajectory_ref_post_inject'),
    ]

    # FDI sees the post-inject signals — identical to sweep mode.
    # Luenberger: needs actual torque applied to plant (post-inject).
    # Admittance inversion: needs actual reference seen by trajectory ctrl.
    fdi_remaps = [
        ('/torque_raw',     '/torque_post_inject'),
        ('/trajectory_ref', '/trajectory_ref_post_inject'),
    ]

    return LaunchDescription([

        # ── Robot state publisher ────────────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}],
            remappings=rsp_remaps,
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(description_pkg, 'rviz', 'exo_display.rviz')],
        ),

        # ── Plant dynamics ───────────────────────────────────────────
        Node(
            package='exoskeleton_dynamics',
            executable='exo_dynamics',
            name='dynamics',
            output='screen',
            remappings=dynamics_remaps,
        ),

        # ── External wrench input (parameters driven by scheduler) ───
        Node(
            package='exoskeleton_utils',
            executable='external_wrench_pub',
            name='input',
            output='screen',
            parameters=[{
                'frequency': 0.05,
                'f_min':    -12.0,
                'f_max':     0.0,
            }],
        ),

        # ── Admittance controller ────────────────────────────────────
        Node(
            package='exoskeleton_control',
            executable='admittance_controller',
            name='admittance_controller',
            output='screen',
            remappings=admittance_remaps,
        ),

        # ── Trajectory controller ────────────────────────────────────
        Node(
            package='exoskeleton_control',
            executable='trajectory_controller',
            name='trajectory_controller',
            output='screen',
            remappings=trajectory_remaps,
        ),

        # ── Fault framework — transparent passthrough (no fault armed) ─
        # active_channel = -1 by default in fault_framework_params.yaml.
        # Including the framework here ensures the characterisation
        # topology is identical to production; see module docstring.
        Node(
            package='exoskeleton_faults',
            executable='fault_framework',
            name='fault_injector',
            output='screen',
            parameters=[fault_fw_params],
        ),

        # ── FDI node ─────────────────────────────────────────────────
        Node(
            package='exoskeleton_fdi',
            executable='fdi_node',
            name='fdi_node',
            output='screen',
            parameters=[fdi_params],
            remappings=fdi_remaps,
        ),

        # ── Regime scheduler ─────────────────────────────────────────
        Node(
            package='exoskeleton_fdi_characterization',
            executable='regime_scheduler',
            name='regime_scheduler',
            output='screen',
            parameters=[regimes_yaml],
        ),

        # ── Bag recorder ─────────────────────────────────────────────
        bag_record,

    ])
