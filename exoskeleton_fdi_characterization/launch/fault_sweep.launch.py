"""
fault_sweep.launch.py
======================

Records a systematic fault-injection sweep over (channel x type x magnitude).

Topology
--------
    dynamics  ──[/joint_states_raw, /exo_dynamics/tau_ext_theta_raw]──►
       fault_framework  ──[/joint_states, /exo_dynamics/tau_ext_theta]──►
         consumers (controllers, observer)

    trajectory_controller  ──[/torque_raw]──► fault_framework
       ──[/torque_post_inject]──► dynamics (no bridge here)

    admittance_controller  ──[/trajectory_ref_raw]──► fault_framework
       ──[/trajectory_ref_post_inject]──► trajectory_controller

    FDI node  ──► (residuals for calibration)

    fault_orchestrator  ──► sets fault_injector parameters via service

    rosbag2 record  ──► everything (incl. *_raw and *_post_inject)

NOTE — Bridge and state machine are NOT launched. The orchestrator
needs to corrupt the signals freely; the safety stack would react and
distort the residuals.

Output
------
    ./bags/fault_sweep_<timestamp>/        rosbag2 archive
    ./fault_sweep_manifest.json             window-by-window metadata

Usage
-----
    ros2 launch exoskeleton_fdi_characterization fault_sweep.launch.py

Once the orchestrator logs 'SWEEP COMPLETE', press Ctrl+C to stop the
recorder. Default sweep is ~50-60 windows totalling ~50 minutes.
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import datetime


def generate_launch_description():

    description_pkg = get_package_share_directory('exoskeleton_description')
    bringup_pkg     = get_package_share_directory('exoskeleton_bringup')
    char_pkg        = get_package_share_directory('exoskeleton_fdi_characterization')
    fdi_pkg          = get_package_share_directory('exoskeleton_fdi')

    urdf_file = os.path.join(description_pkg, 'urdf', 'assembly_with_hand.urdf')
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    # ── Config files ─────────────────────────────────────────────────
    # SWEEP_CONFIG env var overrides the default for testing:
    #   SWEEP_CONFIG=/tmp/quick_sweep.yaml ros2 launch ... fault_sweep.launch.py
    _default_sweep = os.path.join(char_pkg, 'config', 'fault_sweep.yaml')
    sweep_yaml          = os.environ.get('SWEEP_CONFIG', _default_sweep)
    fault_fw_params     = os.path.join(bringup_pkg, 'config', 'fault_framework_params.yaml')
    fdi_params          = os.path.join(fdi_pkg, 'config', 'fdi_params.yaml')

    # ── Bag output ───────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path  = os.path.join('bags', f'fault_sweep_{timestamp}')

    # RSP reads raw joints so the TF tree stays correct even when ch3 is faulted.
    rsp_remaps = [
        ('/joint_states', '/joint_states_raw'),
    ]

    # ── Topics to record ─────────────────────────────────────────────
    record_topics = [
        # Plant — raw (pre-framework) and canonical (post-framework)
        '/joint_states',
        '/joint_states_raw',
        '/exo_dynamics/tau_ext_theta',
        '/exo_dynamics/tau_ext_theta_raw',
        '/exo_dynamics/debug',
        '/exo_dynamics/ff_terms',
        '/exo_dynamics/external_wrench',
        # Control — raw and post-inject
        # NOTE: /torque and /trajectory_ref are NOT published in sweep
        # topology (dynamics/admittance publish to *_raw, framework
        # outputs post_inject).  Recording them would yield empty topics.
        '/torque_raw',
        '/torque_post_inject',
        '/trajectory_ref_raw',
        '/trajectory_ref_post_inject',
        '/admittance/debug',
        '/traj_ctrl/debug',
        # FDI
        '/fdi/debug',
        '/fdi/diagnosis',
        '/fdi/residuals',
        # Fault framework ground truth (channel, type, magnitude, delta)
        '/fault_injector/status',
    ]

    bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', bag_path] + record_topics,
        output='screen',
    )

    # ════════════════════════════════════════════════════════════════
    # Remaps for the fault framework topology (no bridge in path)
    # ════════════════════════════════════════════════════════════════

    # Dynamics: producers go to *_raw so framework can intercept.
    dynamics_remaps = [
        ('/joint_states',                '/joint_states_raw'),
        ('/exo_dynamics/tau_ext_theta',  '/exo_dynamics/tau_ext_theta_raw'),
        # No bridge: dynamics consumes torque from framework directly.
        ('/torque',                      '/torque_post_inject'),
    ]

    # Admittance publishes on _raw, reads canonical sensor topics.
    admittance_remaps = [
        ('/trajectory_ref', '/trajectory_ref_raw'),
    ]

    # Trajectory publishes on _raw, reads framework's reference output.
    trajectory_remaps = [
        ('/torque',          '/torque_raw'),
        ('/trajectory_ref',  '/trajectory_ref_post_inject'),
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

        # ── External wrench input ────────────────────────────────────
        # During the sweep we hold a single, mid-range regime so the
        # only thing changing is the injected fault. (The baseline
        # multi-regime varies the input; the sweep keeps it constant.)
        Node(
            package='exoskeleton_utils',
            executable='external_wrench_pub',
            name='input',
            output='screen',
            parameters=[{
                'frequency': 0.10,
                'f_min':    -10.0,
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

        # ── Fault Framework (multi-channel proxy) ────────────────────
        Node(
            package='exoskeleton_faults',
            executable='fault_framework',
            name='fault_injector',
            output='screen',
            parameters=[fault_fw_params],
        ),

        # ── FDI node ─────────────────────────────────
        # Reads post-injection topics to detect faults on ch0 (tau_ext)
        # and ch3 (encoder). Uses admittance inversion + Luenberger.
        Node(
            package='exoskeleton_fdi',
            executable='fdi_node',
            name='fdi_node',
            output='screen',
            parameters=[fdi_params],
            remappings=[
                ('/torque_raw',        '/torque_post_inject'),
                ('/trajectory_ref',    '/trajectory_ref_post_inject'),
            ],
        ),

        # ── Fault orchestrator ───────────────────────────────
        # ── Fault orchestrator ───────────────────────────────────────
        Node(
            package='exoskeleton_fdi_characterization',
            executable='fault_orchestrator',
            name='fault_orchestrator',
            output='screen',
            parameters=[sweep_yaml],
        ),

        # ── Bag recorder ─────────────────────────────────────────────
        bag_record,

    ])