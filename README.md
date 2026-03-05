This is the support material for the work **Graph-Fused Vision-Language-Action Models for Semantically Safe Dual-Robot Control via Control Barrier Functions**

## Control Barrier Function (CBF) Module

The project includes a CBF-based safety filter for obstacle avoidance during dual-robot arm manipulation. The CBF filter minimally modifies nominal (VLA-predicted) actions to ensure collision-free motion.

### Features

- **Obstacle avoidance**: Keeps end-effectors at safe distance from obstacles (from point cloud or detected blocks)
- **Inter-arm collision avoidance**: Maintains minimum distance between left and right arms
- **Workspace bounds**: Enforces Cartesian workspace limits
- **CBF-QP formulation**: Solves a quadratic program to find the closest safe action to the nominal control

### Usage

CBF is enabled by default. To disable:

```bash
python hardware/demo_build_vla_2d.py --no-cbf
```

### Configuration

In `HardwareConfig.safety`:
- `use_cbf_filter`: Enable/disable CBF (default: True)
- `cbf_obstacle_margin`: Minimum distance from obstacles (default: 0.08 m)
- `cbf_inter_arm_margin`: Minimum distance between arms (default: 0.12 m)

### Programmatic Usage

```python
from hardware import DualArmHardwareInterface, Obstacle, obstacles_from_point_cloud

# Obstacles from point cloud
obstacles = obstacles_from_point_cloud(point_cloud, safety_margin=0.05)

# Update obstacles before execution
hardware.set_cbf_obstacles(obstacles)

# Or pass per-action
hardware.execute_action(action, blocking=True, obstacles=obstacles)
```
