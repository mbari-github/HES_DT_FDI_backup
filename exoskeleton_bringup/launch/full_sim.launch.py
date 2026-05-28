"""
full_sim.launch.py
==================

Complete system launch: plant + control loop + fault framework +
exo_bridge + safety state machine.

Topology
--------
    dynamics --[/joint_states_raw]--> fault_framework
              --[/exo_dynamics/tau_ext_theta_raw]--> fault_framework

    admittance_controller --[/trajectory_ref_raw]--> fault_framework
    trajectory_controller --[/torque_raw]--> fault_framework

    fault_framework re-publishes on canonical topics:
      /joint_states
      /exo_dynamics/tau_ext_theta
      /trajectory_ref_post_inject  -->  exo_bridge  -->  /trajectory_ref
      /torque_post_inject           -->  exo_bridge  -->  /torque  -->  dynamics

    Robot State Publisher reads /joint_states_raw (always correct TF tree).
    RViz reads /joint_states (canonical, shows faulted state when active).

    The exo_bridge subscribes natively to /trajectory_ref_post_inject,
    /torque_post_inject, /joint_states and /exo_dynamics/tau_ext_theta —
    no remaps required on the bridge node.

Runtime fault injection (no relaunch needed):
    ros2 param set /fault_injector active_channel <0|1|2|3|-1>
    ros2 param set /fault_injector fault_type <type>
    ros2 param set /fault_injector fault_magnitude <m>
    ros2 param set /fault_injector fault_active true
"""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    description_pkg = get_package_share_directory('exoskeleton_description')
    launch_pkg      = get_package_share_directory('exoskeleton_bringup')

    urdf_file = os.path.join(description_pkg, 'urdf', 'assembly_with_hand.urdf')
    with open(urdf_file, 'r') as infp:
        robot_description_content = infp.read()

    safety_params_file      = os.path.join(launch_pkg, 'config', 'safety_params.yaml')
    exo_bridge_params_file  = os.path.join(launch_pkg, 'config', 'exo_bridge_params.yaml')
    dynamics_params_file    = os.path.join(launch_pkg, 'config', 'dynamics_params.yaml')
    fault_framework_params  = os.path.join(launch_pkg, 'config', 'fault_framework_params.yaml')

    # ============================================================
    # FAULT INJECTION INITIAL CONFIGURATION
    # These values are passed as initial ROS 2 parameters.
    # All of them can be changed at runtime via ros2 param set.
    # ============================================================
    # active_channel:
    #   -1 -> all passthrough (default)
    #    0 -> /exo_dynamics/tau_ext_theta  (load cell)
    #    1 -> /trajectory_ref              (admittance reference)
    #    2 -> /torque                      (controller output)
    #    3 -> /joint_states                (encoder feedback)

    FAULT_CHANNEL   = -1        # start with all channels in passthrough
    FAULT_TYPE      = 'offset'
    FAULT_MAGNITUDE = 0.1
    FAULT_ACTIVE    = False
    NOISE_STD       = 0.1
    SPIKE_DURATION  = 0.05
    TARGET_INDEX    = 0
    FAULT_JS_FIELD  = 'position'
    JOINT_NAME      = 'rev_crank'

    # ============================================================
    # FIXED REMAPS — no FAULT_CHANNEL conditional
    # ============================================================

    # RSP reads raw joints so TF is never corrupted by a ch3 fault.
    rsp_remaps = [
        ('/joint_states', '/joint_states_raw'),
    ]

    # Dynamics: publish all producer signals on *_raw so the framework
    # can intercept them. /torque is NOT remapped — dynamics reads it
    # from the bridge (which publishes /torque after safety clamping).
    dynamics_remaps = [
        ('/joint_states',                '/joint_states_raw'),
        ('/exo_dynamics/tau_ext_theta',  '/exo_dynamics/tau_ext_theta_raw'),
    ]

    # Admittance: publish reference on /trajectory_ref_raw so the
    # framework intercepts it. Canonical sensor reads need no remap.
    admittance_remaps = [
        ('/trajectory_ref', '/trajectory_ref_raw'),
    ]

    # Trajectory: publish torque on /torque_raw so the framework
    # intercepts it. Reads /trajectory_ref from the bridge output.
    trajectory_remaps = [
        ('/torque', '/torque_raw'),
    ]

    # Bridge: subscribes natively to /trajectory_ref_post_inject,
    # /torque_post_inject, /joint_states, /exo_dynamics/tau_ext_theta.
    # No remaps required.

    # ============================================================
    # NODES — started immediately
    # ============================================================
    nodes_immediate = [

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}],
            remappings=rsp_remaps,
        ),

        Node(
            package='exoskeleton_dynamics',
            executable='exo_dynamics',
            name='dynamics',
            output='screen',
            parameters=[dynamics_params_file],
            remappings=dynamics_remaps,
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(description_pkg, 'rviz', 'exo_display.rviz')],
        ),

        Node(
            package='exoskeleton_utils',
            executable='external_wrench_pub',
            name='input',
            output='screen',
        ),

        Node(
            package='exoskeleton_control',
            executable='admittance_controller',
            name='admittance_controller',
            output='screen',
            remappings=admittance_remaps,
        ),

        Node(
            package='exoskeleton_control',
            executable='trajectory_controller',
            name='trajectory_controller',
            output='screen',
            remappings=trajectory_remaps,
        ),

        Node(
            package='exoskeleton_supervision',
            executable='exo_bridge',
            name='exo_bridge',
            output='screen',
            parameters=[exo_bridge_params_file],
        ),

        Node(
            package='exoskeleton_faults',
            executable='fault_framework',
            name='fault_injector',
            output='screen',
            parameters=[
                fault_framework_params,
                {
                    'active_channel':  FAULT_CHANNEL,
                    'fault_active':    FAULT_ACTIVE,
                    'fault_type':      FAULT_TYPE,
                    'fault_magnitude': FAULT_MAGNITUDE,
                    'noise_std':       NOISE_STD,
                    'spike_duration':  SPIKE_DURATION,
                    'target_index':    TARGET_INDEX,
                    'fault_js_field':  FAULT_JS_FIELD,
                    'joint_name':      JOINT_NAME,
                    'publish_rate':    200.0,
                },
            ],
        ),
    ]

    # ============================================================
    # NODES — delayed start (5 s) to let the bridge and dynamics
    # settle before the safety state machine begins monitoring
    # ============================================================
    nodes_delayed = [

        Node(
            package='exoskeleton_safety_manager',
            executable='state_machine_node',
            name='state_machine',
            output='screen',
            parameters=[safety_params_file],
        ),
    ]

    return LaunchDescription(
        nodes_immediate
        + [
            TimerAction(
                period=5.0,
                actions=nodes_delayed,
            )
        ]
    )
