"""
multi_channel_injection.launch.py
==================================

Launch the digital twin (plant + control loop + multi-channel
fault injector) without bridge or state machine.

This is the reference launch for FDI characterization (Phase 0) and
fault injection sweeps (Phase 0 / Phase 6). The fault_framework runs
permanently as a transparent proxy on all four channels; faults are
armed at runtime via:

    ros2 param set /fault_injector active_channel <0|1|2|3|-1>
    ros2 param set /fault_injector fault_type <type>
    ros2 param set /fault_injector fault_magnitude <m>
    ros2 param set /fault_injector fault_active true

No relaunch is ever required to switch channel. No remap is conditional
on a FAULT_CHANNEL constant.

Topology
--------
    dynamics --[/joint_states_raw]--> fault_framework
              --[/exo_dynamics/tau_ext_theta_raw]--> fault_framework

    admittance_controller --[/trajectory_ref_raw]--> fault_framework
    trajectory_controller --[/torque_raw]--> fault_framework

    fault_framework re-publishes on canonical topics:
      /joint_states
      /exo_dynamics/tau_ext_theta
      /trajectory_ref_post_inject     (consumed by bridge if launched)
      /torque_post_inject             (consumed by bridge if launched)

Note: the bridge is NOT launched here. For tests with bridge+SM use
multi_channel_injection_with_safety.launch.py.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ── Package paths ────────────────────────────────────────────────
    description_pkg = get_package_share_directory('exoskeleton_description')
    bringup_pkg     = get_package_share_directory('exoskeleton_bringup')

    # ── URDF ─────────────────────────────────────────────────────────
    urdf_file = os.path.join(
        description_pkg, 'urdf', 'assembly_with_hand.urdf'
    )
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    # ── Config ───────────────────────────────────────────────────────
    fault_framework_params = os.path.join(
        bringup_pkg, 'config', 'fault_framework_params.yaml'
    )

    # ════════════════════════════════════════════════════════════════
    # FIXED REMAPS — no FAULT_CHANNEL conditional anymore
    # ════════════════════════════════════════════════════════════════

    # Dynamics: publishes producer signals on the *_raw topics so the
    # framework can intercept them.
    dynamics_remaps = [
        ('/joint_states',                '/joint_states_raw'),
        ('/exo_dynamics/tau_ext_theta',  '/exo_dynamics/tau_ext_theta_raw'),
    ]

    # Admittance controller publishes on /trajectory_ref_raw (existing
    # convention) and consumes canonical sensor topics.
    admittance_remaps = [
        ('/trajectory_ref', '/trajectory_ref_raw'),
    ]

    # Trajectory controller publishes on /torque_raw (existing
    # convention). Since the bridge is NOT in this launch, it reads
    # the framework output directly via /trajectory_ref_post_inject.
    trajectory_remaps = [
        ('/torque',          '/torque_raw'),
        ('/trajectory_ref',  '/trajectory_ref_post_inject'),
    ]

    # Dynamics consumes the framework torque output directly (no bridge).
    dynamics_remaps.append(
        ('/torque', '/torque_post_inject'),
    )

    # No remaps for RViz, observers, FDI: they read canonical names.

    # RSP reads from /joint_states_raw so the TF tree stays correct even
    # when channel 3 is faulted. The fault_framework publishes the (possibly
    # corrupted) data to /joint_states; RSP must bypass that.
    rsp_remaps = [
        ('/joint_states', '/joint_states_raw'),
    ]

    return LaunchDescription([

        # ── Robot State Publisher ────────────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}],
            remappings=rsp_remaps,
        ),

        # ── Plant dynamics ───────────────────────────────────────────
        Node(
            package='exoskeleton_dynamics',
            executable='exo_dynamics',
            name='dynamics',
            output='screen',
            remappings=dynamics_remaps,
        ),

        # ── RViz ─────────────────────────────────────────────────────
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=[
                '-d',
                os.path.join(description_pkg, 'rviz', 'exo_display.rviz'),
            ],
        ),

        # ── External wrench input (sinusoidal) ───────────────────────
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

        # ── Admittance controller (outer loop) ───────────────────────
        Node(
            package='exoskeleton_control',
            executable='admittance_controller',
            name='admittance_controller',
            output='screen',
            remappings=admittance_remaps,
        ),

        # ── Trajectory controller (inner loop) ───────────────────────
        Node(
            package='exoskeleton_control',
            executable='trajectory_controller',
            name='trajectory_controller',
            output='screen',
            remappings=trajectory_remaps,
        ),

        # ── Fault Framework (permanent multi-channel proxy) ──────────
        Node(
            package='exoskeleton_faults',
            executable='fault_framework',
            name='fault_injector',
            output='screen',
            parameters=[fault_framework_params],
        ),

    ])