# ROS2 Integration for Dual-Arm Hardware

This directory contains ROS2 nodes and launch files for communicating with the dual-arm robot system via ROS2.

## Overview

The ROS2 integration provides:
- **ROS2 Bridge Node**: Communicates with hardware and publishes/subscribes to ROS2 topics
- **Demo Nodes**: ROS2-based demonstrations for tower building and VLA letter building
- **Launch Files**: Easy startup scripts for running demos

## Installation

### Prerequisites

1. **ROS2** (Humble or later):
   ```bash
   # Install ROS2 (Ubuntu)
   sudo apt install ros-humble-desktop
   source /opt/ros/humble/setup.bash
   ```

2. **ROS2 Dependencies**:
   ```bash
   sudo apt install ros-humble-cv-bridge \
                     ros-humble-sensor-msgs-py \
                     python3-opencv
   ```

3. **Python Dependencies**:
   ```bash
   pip install rclpy numpy opencv-python
   ```

### Building the Package

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws/src

# Clone or copy this package
# (Assuming the hardware folder is already in your project)

# Build
cd ~/ros2_ws
colcon build --packages-select dualarmvla_hardware
source install/setup.bash
```

## Usage

### Starting the Bridge Node

The bridge node connects to hardware and provides ROS2 interfaces:

```bash
# Using launch file
ros2 launch dualarmvla_hardware dual_arm_bridge.launch.py \
    ur5e_ip:=192.168.1.101 \
    ur10e_ip:=192.168.1.102 \
    robotiq_port:=/dev/ttyUSB0 \
    barrett_ip:=192.168.1.103

# Or directly
ros2 run dualarmvla_hardware dual_arm_ros2_bridge
```

### Running Tower Building Demo

```bash
# Using launch file (starts both bridge and demo)
ros2 launch dualarmvla_hardware demo_tower.launch.py

# Or run separately
# Terminal 1: Start bridge
ros2 run dualarmvla_hardware dual_arm_ros2_bridge

# Terminal 2: Run demo
ros2 run dualarmvla_hardware demo_tower_ros2
```

### Running VLA Letters Demo

```bash
# Using launch file
ros2 launch dualarmvla_hardware demo_vla_letters.launch.py

# Or run separately
ros2 run dualarmvla_hardware demo_vla_letters_ros2
```

## ROS2 Topics

### Published Topics

- `/dual_arm/left_arm/pose` (geometry_msgs/PoseStamped): UR5e end-effector pose
- `/dual_arm/right_arm/pose` (geometry_msgs/PoseStamped): UR10e end-effector pose
- `/dual_arm/left_arm/joint_states` (std_msgs/Float64MultiArray): UR5e joint states
- `/dual_arm/right_arm/joint_states` (std_msgs/Float64MultiArray): UR10e joint states
- `/dual_arm/left_arm/gripper_state` (std_msgs/Float64MultiArray): Robotiq gripper state
- `/dual_arm/right_arm/gripper_state` (std_msgs/Float64MultiArray): Barrett Hand state
- `/dual_arm/camera/rgb` (sensor_msgs/Image): RGB image from camera
- `/dual_arm/camera/depth` (sensor_msgs/Image): Depth image from camera
- `/dual_arm/camera/pointcloud` (sensor_msgs/PointCloud2): Point cloud from camera
- `/dual_arm/status` (std_msgs/String): System status

### Subscribed Topics

- `/dual_arm/action` (std_msgs/Float64MultiArray): 14-dimensional dual-arm action
- `/dual_arm/left_arm/command` (std_msgs/Float64MultiArray): Left arm only command (7D)
- `/dual_arm/right_arm/command` (std_msgs/Float64MultiArray): Right arm only command (7D)
- `/dual_arm/emergency_stop` (std_msgs/Bool): Emergency stop command

## ROS2 Services

- `/dual_arm/connect` (std_srvs/Trigger): Connect to hardware
- `/dual_arm/disconnect` (std_srvs/Trigger): Disconnect from hardware
- `/dual_arm/reset` (std_srvs/Trigger): Reset robots to home position
- `/dual_arm/get_observation` (std_srvs/Trigger): Get current observation

## Example: Sending Actions via ROS2

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

class ActionPublisher(Node):
    def __init__(self):
        super().__init__('action_publisher')
        self.pub = self.create_publisher(
            Float64MultiArray, '/dual_arm/action', 10
        )
    
    def send_action(self, action):
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.pub.publish(msg)

# Usage
rclpy.init()
node = ActionPublisher()
action = np.array([
    0.4, 0.0, 0.3, 0.0, 0.0, 0.0,  # UR5e pose
    1.0,  # Robotiq gripper open
    0.4, 0.0, 0.3, 0.0, 0.0, 0.0,  # UR10e pose
    1.0,  # Barrett Hand open
])
node.send_action(action)
```

## Integration with VLA Models

The ROS2 bridge can be used with VLA models by:
1. Subscribing to camera topics for observations
2. Publishing actions to `/dual_arm/action`
3. Using services for connection management

See `hardware/vla_integration.py` for VLA model integration examples.

