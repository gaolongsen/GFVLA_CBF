<p align="center">
  <h1 align="center">Graph-Fused Vision-Language-Action Models for Semantically Safe Dual-Robot Control via Control Barrier Functions</h1>
  <p align="center">
  </p>
  <p align="center">
    <a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
    <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-blue" alt="Citation"></a>
    <a href="#control-barrier-function-cbf-module"><img src="https://img.shields.io/badge/Safety-CBF-green" alt="CBF"></a>
  </p>
</p>


---

Official code and support material for our work on **semantically safe dual-robot manipulation** using Graph-Fused Vision-Language-Action (GF-VLA) models with Control Barrier Functions (CBFs). This framework enables language-conditioned dual-arm control while guaranteeing collision-free motion through real-time CBF-based safety filtering.

<p align="center">
  <img src="https://github.com/gaolongsen/picx-images-hosting/raw/master/block_diagram_1.41ylrcxitx.webp" alt="GF-VLA Overview" width="80%"/>
    <img src="https://github.com/gaolongsen/picx-images-hosting/raw/master/block_diagram_2.26m0yql38f.webp" alt="GF-VLA Overview" width="80%"/>
</p>






---

## ✨ Highlights

- **Dual-arm VLA control** — UR5e + UR10e coordination for manipulation tasks
- **CBF safety filter** — Minimal modification of nominal actions for guaranteed obstacle avoidance
- **Vision-language grounding** — Point cloud + language instruction for semantic understanding
- **Real-time QP solver** — Efficient CBF-QP formulation for online safety filtering

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/GFVLA_CBF.git
cd GFVLA_CBF

# Install dependencies
pip install -r requirement.txt
```

### Run Dual-Arm Demo

```bash
# Build "VLA" letters with Jenga blocks (CBF enabled by default)
python hardware/demo_build_vla_2d.py --ur5e-ip 192.168.1.101 --ur10e-ip 192.168.1.102

# Simulation mode (no hardware)
python hardware/demo_build_vla_2d.py --simulation
```

---

## 🛡️ Control Barrier Function (CBF) Module

The CBF module provides **safety filtering** that minimally modifies VLA-predicted actions to ensure collision-free motion during dual-robot manipulation.

### Features

| Feature | Description |
|---------|-------------|
| **Obstacle avoidance** | Keeps end-effectors at safe distance from obstacles (point cloud / detected blocks) |
| **Inter-arm collision** | Maintains minimum distance between left and right arms |
| **Workspace bounds** | Enforces Cartesian workspace limits |
| **CBF-QP formulation** | Solves QP to find closest safe action to nominal control |

### Usage

CBF is **enabled by default**. To disable:

```bash
python hardware/demo_build_vla_2d.py --no-cbf
```

### Configuration

In `HardwareConfig.safety`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_cbf_filter` | `True` | Enable/disable CBF filter |
| `cbf_obstacle_margin` | `0.08` m | Minimum distance from obstacles |
| `cbf_inter_arm_margin` | `0.12` m | Minimum distance between arms |

### Programmatic Usage

```python
from hardware import DualArmHardwareInterface, Obstacle, obstacles_from_point_cloud

# Create hardware interface
hardware = DualArmHardwareInterface(config)

# Obstacles from point cloud (e.g., depth camera)
obstacles = obstacles_from_point_cloud(point_cloud, safety_margin=0.05)

# Option 1: Update obstacles globally
hardware.set_cbf_obstacles(obstacles)

# Option 2: Pass per-action
hardware.execute_action(action, blocking=True, obstacles=obstacles)
```

---

## 📁 Project Structure

```
GFVLA_CBF/
├── hardware/           # Dual-arm hardware & CBF
│   ├── cbf.py          # Control Barrier Function module
│   ├── dual_arm_hardware.py
│   ├── vla_integration.py
│   └── demo_build_vla_2d.py
├── models/             # VLA model definitions
├── vla/                # VLA training & datasets
├── scripts/            # Training scripts
└── assets/             # Figures & resources
```

---

## 📄 License

This project is released for research purposes. See individual files for license details.
