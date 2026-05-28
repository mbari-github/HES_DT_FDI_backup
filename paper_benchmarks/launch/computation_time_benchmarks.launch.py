from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directories
    description_pkg = get_package_share_directory('exoskeleton_description')
    benchmark_pkg = get_package_share_directory('paper_benchmarks')
    
    # Use URDF from paper_benchmarks package
    urdf_file = os.path.join(benchmark_pkg, 'urdf', 'assembly_with_hand.urdf')
    
    with open(urdf_file, 'r') as infp:
        robot_description_content = infp.read()
    
    # Launch arguments
    profiling_output_file_arg = DeclareLaunchArgument(
        'profiling_output_file',
        default_value='/tmp/benchmark_timing.csv',
        description='Output CSV file for profiling data'
    )
    
    profiling_detail_arg = DeclareLaunchArgument(
        'profiling_detail',
        default_value='basic',
        description='Profiling detail level: basic or detailed'
    )
    
    auto_stop_after_sec_arg = DeclareLaunchArgument(
        'auto_stop_after_sec',
        default_value='52',
        description='Auto-stop after N seconds (52 = 50000 measured + 2000 warmup at dt=0.001)'
    )
    
    return LaunchDescription([
        profiling_output_file_arg,
        profiling_detail_arg,
        auto_stop_after_sec_arg,
        
        # Robot State Publisher — broadcasts TF from URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}]
        ),
        
        # Reduced dynamics simulation with profiling enabled
        Node(
            package='paper_benchmarks',
            executable='benchmark_dynamics',
            name='dynamics',
            output='screen',
            parameters=[{
                'urdf_path': urdf_file,
                'profiling_enabled': True,
                'profiling_output_file': LaunchConfiguration('profiling_output_file'),
                'profiling_detail': LaunchConfiguration('profiling_detail'),
                'auto_stop_after_sec': LaunchConfiguration('auto_stop_after_sec'),
                'dt': 0.001,
                'max_nfev': 150,
                'publish_dt': 0.005,
                'external_wrench_enable': False,
                'ce_force_enable': False,
                'profiling_warmup': 2000,
            }]
        ),
        
    ])
