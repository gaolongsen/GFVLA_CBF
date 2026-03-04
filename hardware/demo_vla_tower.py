"""
Dual-arm demonstration using VLA model: Build a 3D tower.

This script uses the trained GF-VLA model to control dual-arm robots
to build a tower using Jenga blocks, based on language instructions.
"""

import numpy as np
import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models import load_vla
from hardware import DualArmHardwareInterface, HardwareConfig, CameraInterface
from hardware.vla_integration import VLAHardwareController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run VLA-based tower building."""
    parser = argparse.ArgumentParser(
        description="Dual-arm VLA demonstration: Build a 3D tower with Jenga blocks"
    )
    
    # Model arguments
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained VLA model checkpoint'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default='',
        help='HuggingFace token (if loading from HF Hub)'
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='0',
        help='CUDA device ID'
    )
    parser.add_argument(
        '--predict-mode',
        type=str,
        default='diff+ar',
        choices=['diff', 'ar', 'diff+ar'],
        help='Prediction mode'
    )
    parser.add_argument(
        '--cfg-scale',
        type=float,
        default=1.5,
        help='Classifier-free guidance scale'
    )
    parser.add_argument(
        '--ddim-steps',
        type=int,
        default=10,
        help='Number of DDIM steps'
    )
    parser.add_argument(
        '--use-diff',
        type=int,
        default=1,
        help='Use diffusion (1) or not (0)'
    )
    parser.add_argument(
        '--use-ar',
        type=int,
        default=1,
        help='Use autoregressive (1) or not (0)'
    )
    parser.add_argument(
        '--load-pointcloud',
        type=int,
        default=1,
        help='Load point cloud (1) or not (0)'
    )
    parser.add_argument(
        '--use-robot-state',
        type=int,
        default=1,
        help='Use robot state (1) or not (0)'
    )
    parser.add_argument(
        '--slow-fast-ratio',
        type=int,
        default=4,
        help='Ratio between slow and fast image updates'
    )
    parser.add_argument(
        '--training-diffusion-steps',
        type=int,
        default=100,
        help='Number of diffusion steps used during training'
    )
    parser.add_argument(
        '--llm-middle-layer',
        type=int,
        default=32,
        help='LLM middle layer index'
    )
    parser.add_argument(
        '--action-chunk',
        type=int,
        default=1,
        help='Action chunk size'
    )
    parser.add_argument(
        '--lang-subgoals-exist',
        type=int,
        default=0,
        help='Language subgoals exist (1) or not (0)'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--ur5e-ip',
        type=str,
        default='192.168.1.101',
        help='UR5e robot IP address'
    )
    parser.add_argument(
        '--ur10e-ip',
        type=str,
        default='192.168.1.102',
        help='UR10e robot IP address'
    )
    parser.add_argument(
        '--robotiq-port',
        type=str,
        default='/dev/ttyUSB0',
        help='Robotiq gripper serial port'
    )
    parser.add_argument(
        '--barrett-ip',
        type=str,
        default='192.168.1.103',
        help='Barrett Hand IP address'
    )
    parser.add_argument(
        '--camera-type',
        type=str,
        default='realsense',
        choices=['realsense', 'kinect', 'simulation'],
        help='Camera type for observations'
    )
    
    # Task arguments
    parser.add_argument(
        '--instruction',
        type=str,
        default='Build a tower by stacking Jenga blocks vertically',
        help='Language instruction for the task'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=50,
        help='Maximum number of steps per episode'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Run in simulation mode (no actual hardware connection)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Dual-Arm VLA Tower Building Demonstration")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Instruction: {args.instruction}")
    logger.info(f"Prediction mode: {args.predict_mode}")
    logger.info(f"Hardware simulation: {args.simulation}")
    logger.info("=" * 60)
    
    # Determine prediction mode
    if args.use_diff == 1 and args.use_ar == 0:
        predict_mode = 'diff'
    elif args.use_diff == 0 and args.use_ar == 1:
        predict_mode = 'ar'
    elif args.use_diff == 1 and args.use_ar == 1:
        predict_mode = 'diff+ar'
    else:
        predict_mode = 'ar'
    
    # Load VLA model
    logger.info("Loading VLA model...")
    try:
        model = load_vla(
            args.model_path,
            load_for_training=False,
            future_action_window_size=15,
            hf_token=args.hf_token,
            use_diff=args.use_diff == 1,
            diffusion_steps=args.training_diffusion_steps,
            llm_middle_layer=args.llm_middle_layer,
            training_mode='async',
            load_pointcloud=args.load_pointcloud == 1,
            pointcloud_pos='slow',
            action_chunk=args.action_chunk,
            load_state=args.use_robot_state == 1,
            lang_subgoals_exist=args.lang_subgoals_exist == 1,
        )
        model.to(f'cuda:{args.cuda}').eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return 1
    
    # Initialize hardware
    config = HardwareConfig()
    config.ur5e.ip_address = args.ur5e_ip
    config.ur10e.ip_address = args.ur10e_ip
    config.robotiq.port = args.robotiq_port
    config.barrett.ip_address = args.barrett_ip
    
    hardware = DualArmHardwareInterface(config)
    
    # Initialize camera
    camera = CameraInterface(camera_type=args.camera_type)
    if not args.simulation:
        if not camera.connect():
            logger.warning("Failed to connect to camera")
            camera = None
    else:
        camera.connect()  # Connect in simulation mode too
    
    # Connect hardware
    if not args.simulation:
        logger.info("Connecting to hardware...")
        if not hardware.connect():
            logger.error("Failed to connect to hardware")
            return 1
    else:
        hardware.connected = True  # Set flag for simulation
    
    try:
        # Create VLA controller
        controller = VLAHardwareController(
            model=model,
            hardware=hardware,
            camera=camera,
            use_pointcloud=args.load_pointcloud == 1,
            use_robot_state=args.use_robot_state == 1,
            action_dim=14,  # Dual-arm
            predict_mode=predict_mode,
            cfg_scale=args.cfg_scale,
            ddim_steps=args.ddim_steps,
            slow_fast_ratio=args.slow_fast_ratio,
        )
        
        # Run episodes
        success_count = 0
        for episode in range(args.num_episodes):
            logger.info(f"Episode {episode + 1}/{args.num_episodes}")
            
            results = controller.run_episode(
                instruction=args.instruction,
                max_steps=args.max_steps,
            )
            
            if results['success']:
                success_count += 1
            
            logger.info(f"Episode {episode + 1} completed: success={results['success']}, steps={results['steps']}")
        
        logger.info(f"All episodes completed. Success rate: {success_count}/{args.num_episodes}")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        if not args.simulation:
            hardware.emergency_stop()
        return 1
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        if not args.simulation:
            hardware.emergency_stop()
        return 1
    finally:
        if camera is not None:
            camera.disconnect()
        if not args.simulation:
            hardware.disconnect()
            logger.info("Hardware disconnected")


if __name__ == '__main__':
    exit(main())

