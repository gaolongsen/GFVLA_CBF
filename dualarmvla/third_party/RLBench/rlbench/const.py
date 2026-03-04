from pyrep.robots.arms.jaco import Jaco
from pyrep.robots.arms.mico import Mico
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.robots.end_effectors.jaco_gripper import JacoGripper
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.robots.end_effectors.robotiq85_gripper import Robotiq85Gripper

# Import Barrett Hand gripper
try:
    from rlbench.end_effectors.barrett_hand_gripper import BarrettHandGripper
except ImportError:
    # Fallback: if import fails, use Robotiq85Gripper as placeholder
    # User should ensure Barrett Hand model is available in the scene
    BarrettHandGripper = Robotiq85Gripper

# Import UR5e and UR10e - they use the same class as UR5 since they share the same kinematics
# UR5e and UR10e are compatible with UR5 class from PyRep
UR5e = UR5
UR10e = UR5


colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('maroon', (0.5, 0.0, 0.0)),
    ('lime', (0.0, 1.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]

# Arms from PyRep need to be modified to include a wrist camera.
# Currently, only the arms/grippers below are supported.
SUPPORTED_ROBOTS = {
    'panda': (Panda, PandaGripper, 7),
    'jaco': (Jaco, JacoGripper, 6),
    'mico': (Mico, MicoGripper, 6),
    'sawyer': (Sawyer, BaxterGripper, 7),
    'ur5': (UR5, Robotiq85Gripper, 6),
    'ur5e': (UR5e, Robotiq85Gripper, 6),  # UR5e with Robotiq Gripper
    'ur10e': (UR10e, BarrettHandGripper, 6),  # UR10e with WAM Barrett Hand (BH282)
}