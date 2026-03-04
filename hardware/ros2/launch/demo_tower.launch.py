"""
ROS2 launch file for tower building demonstration.

This launch file starts both the bridge and the tower builder nodes.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    """Generate launch description for tower building demo."""
    
    # Launch arguments
    ur5e_ip_arg = DeclareLaunchArgument('ur5e_ip', default_value='192.168.1.101')
    ur10e_ip_arg = DeclareLaunchArgument('ur10e_ip', default_value='192.168.1.102')
    robotiq_port_arg = DeclareLaunchArgument('robotiq_port', default_value='/dev/ttyUSB0')
    barrett_ip_arg = DeclareLaunchArgument('barrett_ip', default_value='192.168.1.103')
    camera_type_arg = DeclareLaunchArgument('camera_type', default_value='realsense')
    start_bridge_arg = DeclareLaunchArgument('start_bridge', default_value='true')
    
    # Bridge node
    bridge_node = Node(
        package='dualarmvla_hardware',
        executable='dual_arm_ros2_bridge',
        name='dual_arm_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_bridge')),
        parameters=[{
            'ur5e_ip': LaunchConfiguration('ur5e_ip'),
            'ur10e_ip': LaunchConfiguration('ur10e_ip'),
            'robotiq_port': LaunchConfiguration('robotiq_port'),
            'barrett_ip': LaunchConfiguration('barrett_ip'),
            'camera_type': LaunchConfiguration('camera_type'),
        }]
    )
    
    # Tower builder node
    tower_builder_node = Node(
        package='dualarmvla_hardware',
        executable='demo_tower_ros2',
        name='tower_builder',
        output='screen',
    )
    
    return LaunchDescription([
        ur5e_ip_arg,
        ur10e_ip_arg,
        robotiq_port_arg,
        barrett_ip_arg,
        camera_type_arg,
        start_bridge_arg,
        bridge_node,
        tower_builder_node,
    ])

