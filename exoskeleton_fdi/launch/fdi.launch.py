"""
fdi.launch.py
==============

Launch the digital twin (plant + control loop + fault framework + FDI node)
without bridge or state machine.

Topology
--------
    dynamics --[/joint_states_raw]--> fault_framework
              --[/exo_dynamics/tau_ext_theta_raw]--> fault_framework

    admittance_controller --[/trajectory_ref_raw]--> fault_framework
    trajectory_controller --[/torque_raw]--> fault_framework

    fault_framework re-publishes on canonical topics:
      /joint_states
      /exo_dynamics/tau_ext_theta
      /trajectory_ref_post_inject
      /torque_post_inject

    FDI node reads post-injection topics:
      /joint_states                     (from fault_framework, faulted if ch3)
      /exo_dynamics/tau_ext_theta       (from fault_framework, faulted if ch0)
      /torque_raw  → /torque_post_inject
      /trajectory_ref → /trajectory_ref_post_inject
      /exo_dynamics/ff_terms            (from dynamics)

    Robot State Publisher reads raw joints for correct TF tree:
      /joint_states → /joint_states_raw

Runtime fault injection:
    ros2 param set /fault_injector active_channel 0
    ros2 param set /fault_injector fault_type offset
    ros2 param set /fault_injector fault_magnitude 2.0
    ros2 param set /fault_injector fault_active true

    ros2 param set /fault_injector active_channel 3
    ros2 param set /fault_injector fault_type offset
    ros2 param set /fault_injector fault_magnitude 0.05
    ros2 param set /fault_injector fault_active true
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ── Package paths ────────────────────────────────────────────────
    description_pkg = get_package_share_directory('exoskeleton_description')
    bringup_pkg = get_package_share_directory('exoskeleton_bringup')
    fdi_pkg = get_package_share_directory('exoskeleton_fdi')

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
    fdi_params = os.path.join(fdi_pkg, 'config', 'fdi_params.yaml')

    # ════════════════════════════════════════════════════════════════
    # FIXED REMAPS — topology matches new_injection_testing.launch.py
    # ════════════════════════════════════════════════════════════════

    # Dynamics: publishes on *_raw topics, consumes framework output
    dynamics_remaps = [
        ('/joint_states',                '/joint_states_raw'),
        ('/exo_dynamics/tau_ext_theta',  '/exo_dynamics/tau_ext_theta_raw'),
        ('/torque',                      '/torque_post_inject'),
    ]

    # Admittance controller: publishes on *_raw, consumes canonical sensors
    admittance_remaps = [
        ('/trajectory_ref', '/trajectory_ref_raw'),
    ]

    # Trajectory controller: publishes on *_raw, consumes framework output
    trajectory_remaps = [
        ('/torque',         '/torque_raw'),
        ('/trajectory_ref', '/trajectory_ref_post_inject'),
    ]

    # FDI node: reads canonical topics.
    # /torque_raw is remapped to /torque_post_inject so we read tau_m from
    # the fault framework output (post-injection), matching what dynamics sees.
    # /trajectory_ref is remapped to /trajectory_ref_post_inject for the same
    # reason: this is the reference that trajectory_controller actually consumed.
    fdi_remaps = [
        ('/torque_raw',        '/torque_post_inject'),
        ('/trajectory_ref',    '/trajectory_ref_post_inject'),
    ]

    # Robot State Publisher: reads from /joint_states_raw so the TF tree
    # always reflects the correct closed-chain kinematics, even when ch3
    # (joint_states) is faulted. RViz subscribes to /joint_states (post-fault)
    # for the visual model but relies on TF from the raw joint states.
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

        # ── FDI node ────────────────────────────────────────────────
        Node(
            package='exoskeleton_fdi',
            executable='fdi_node',
            name='fdi_node',
            output='screen',
            parameters=[fdi_params],
            remappings=fdi_remaps,
        ),

    ])
