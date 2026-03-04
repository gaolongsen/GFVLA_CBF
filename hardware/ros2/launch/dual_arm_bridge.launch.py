"""
ROS2 launch file for dual-arm hardware bridge.

This launch file starts the ROS2 bridge node that communicates with
the dual-arm robot hardware (UR5e + UR10e).
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for dual-arm bridge."""
    
    # Launch arguments
    ur5e_ip_arg = DeclareLaunchArgument(
        'ur5e_ip',
        default_value='192.168.1.101',
        description='UR5e robot IP address'
    )
    
    ur10e_ip_arg = DeclareLaunchArgument(
        'ur10e_ip',
        default_value='192.168.1.102',
        description='UR10e robot IP address'
    )
    
    robotiq_port_arg = DeclareLaunchArgument(
        'robotiq_port',
        default_value='/dev/ttyUSB0',
        description='Robotiq gripper serial port'
    )
    
    barrett_ip_arg = DeclareLaunchArgument(
        'barrett_ip',
        default_value='192.168.1.103',
        description='Barrett Hand IP address'
    )
    
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='realsense',
        description='Camera type (realsense, kinect, simulation)'
    )
    
    # Dual-arm bridge node
    bridge_node = Node(
        package='dualarmvla_hardware',
        executable='dual_arm_ros2_bridge',
        name='dual_arm_bridge',
        output='screen',
        parameters=[{
            'ur5e_ip': LaunchConfiguration('ur5e_ip'),
            'ur10e_ip': LaunchConfiguration('ur10e_ip'),
            'robotiq_port': LaunchConfiguration('robotiq_port'),
            'barrett_ip': LaunchConfiguration('barrett_ip'),
            'camera_type': LaunchConfiguration('camera_type'),
        }]
    )
    
    return LaunchDescription([
        ur5e_ip_arg,
        ur10e_ip_arg,
        robotiq_port_arg,
        barrett_ip_arg,
        camera_type_arg,
        bridge_node,
    ])

