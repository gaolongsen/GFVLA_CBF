import os, sys, pathlib
import argparse
import tqdm
import shutil
from termcolor import cprint, colored

from gfvla_env.envs.rlbench_env import RLBenchEnv, RLBenchActionMode, RLBenchObservationConfig
from gfvla_env.helpers.gymnasium import VideoWrapper
from gfvla_env.helpers.common import Logger
from gfvla_env.helpers.graphics import EEpose
import logging
import time
from datetime import datetime

import numpy as np
import pickle

from models import load_vla
import torch
from PIL import Image


def setup_logger(log_dir):
    log_filename = os.path.join(log_dir, "output.log")
    
    logger = logging.getLogger("RLBenchLogger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

def model_load(args):
    model = load_vla(
            args.model_path,
            load_for_training=False,
            future_action_window_size=int(args.model_action_steps),
            hf_token=args.hf_token,
            use_diff = 1,
            diffusion_steps = args.training_diffusion_steps,
            llm_middle_layer = args.llm_middle_layer,
            training_mode = args.training_mode,
            load_pointcloud = args.load_pointcloud,
            pointcloud_pos=args.pointcloud_pos,
            action_chunk=args.action_chunk,
            load_state=args.use_robot_state,
            lang_subgoals_exist=args.lang_subgoals_exist,
            )
    # (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16
    model.to(f'cuda:{args.cuda}').eval()
    return model

def model_predict(args, predict_mode, model, image, prompt, cur_robot_state=None, slow_image=None, point_cloud=None, input_ids = None, slow_latent_embedding=None):
    if predict_mode == 'ar' or predict_mode == 'diff+ar':
        output = model.predict_action(
                image_head_slow = slow_image,
                image_head_fast = image,
                point_cloud = point_cloud,
                instruction = prompt,
                unnorm_key='rlbench',
                cfg_scale = float(args.cfg_scale), 
                use_ddim = True,
                num_ddim_steps = int(args.ddim_steps),
                cur_robot_state = cur_robot_state,
                action_dim = 7,
                predict_mode = predict_mode,
                )
    elif predict_mode == 'diff':
        output = model.fast_system_forward(
                image_head_fast = image,
                point_cloud=point_cloud,
                slow_latent_embedding = slow_latent_embedding,
                input_ids = input_ids,
                unnorm_key = 'rlbench',
                cur_robot_state = cur_robot_state,
                cfg_scale = float(args.cfg_scale), 
                use_ddim = True,
                num_ddim_steps = int(args.ddim_steps),
                action_dim = 7,
                predict_mode = predict_mode,
                )
    return output

def model_predict_slow_latent_embedding(model, prompt, slow_image):
    input_ids, slow_latent_embedding = model.slow_system_forward(
        image_head_slow = slow_image,
        instruction = prompt,
        unnorm_key = 'rlbench',
        )
    return input_ids, slow_latent_embedding

def main(args):
    # Report the arguments
    Logger.log_info(f'Running {colored(__file__, "red")} with arguments:')
    Logger.log_info(f'task name: {args.task_name}')
    Logger.log_info(f'number of episodes: {args.num_episodes}')
    Logger.log_info(f'result directory: {args.result_dir}')
    Logger.log_info(f'replay data directory: {args.replay_data_dir}')
    Logger.log_info(f'exp name: {args.exp_name}')
    Logger.log_info(f'actions steps: {args.model_action_steps}')
    Logger.log_info(f'replay or predict: {args.replay_or_predict}')
    Logger.log_info(f'max steps: {args.max_steps}')
    Logger.log_info(f'cuda used: {args.cuda}')
    cprint('-' * os.get_terminal_size().columns, 'cyan')

    # Check if dual-arm mode is enabled (for GF-VLA)
    use_dual_arm = bool(getattr(args, 'use_dual_arm', 0))
    
    if use_dual_arm:
        # Dual-arm setup: UR5e (Robotiq) + UR10e (Barrett BH282)
        from gfvla_env.envs.dual_arm_env import create_dual_arm_env
        Logger.log_info("Using dual-arm configuration: UR5e (Robotiq) + UR10e (Barrett BH282)")
        env = create_dual_arm_env(
            task_name=args.task_name,
            headless=False,
            robot_setup_left='ur5e',  # UR5e with Robotiq Gripper
            robot_setup_right='ur10e',  # UR10e with Barrett BH282 Hand
        )
        env.launch()
        action_dim = 14  # 2 × (6 DOF + 1 gripper)
    else:
        # Single-arm setup (default)
        action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
        obs_config = RLBenchObservationConfig.single_view_config(camera_name='front', image_size=(224, 224))
        env = RLBenchEnv(
            task_name=args.task_name,
            action_mode=action_mode,
            obs_config=obs_config,
            point_cloud_camera_names=['front'],
            cinematic_record_enabled=True,
            num_points=1024,
            use_point_crop=True,
        )
        action_dim = 7  # 6 DOF + 1 gripper
    env = VideoWrapper(env)
    
    if args.replay_or_predict == 'predict':
        args.result_dir = os.path.join(args.result_dir, 'predict_results')
    elif args.replay_or_predict == 'replay':
        args.result_dir = os.path.join(args.result_dir, 'replay_results')
    
    if args.exp_name is None:
        args.exp_name = args.task_name

    video_dir = os.path.join(
        args.result_dir, args.task_name, args.exp_name, "videos"
    )
    
    recreate_directory(video_dir)
    log_dir = os.path.join(video_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    recreate_directory(log_dir)
    logger = setup_logger(log_dir)

    if int(args.use_diff)==1 and int(args.use_ar)==0:
        predict_mode = 'diff'
    elif int(args.use_diff)==0 and int(args.use_ar)==1:
        predict_mode = 'ar'
    elif int(args.use_diff)==1 and int(args.use_ar)==1:
        predict_mode = 'diff+ar'

    success_num = 0
    # #----------- for model predict
    if args.replay_or_predict == 'predict':
        model = model_load(args)
        episode_length = args.max_steps

    for i in range(args.num_episodes):

        # #----------- for key frames replay
        if args.replay_or_predict == 'replay_key':
            dat = np.load(os.path.join(args.replay_data_dir, args.task_name, f'episode{i}.npy'),allow_pickle = True)
            prompt = dat[0]['language_instruction']
            episode_length = len(dat)

        # #----------- for all frames replay
        if args.replay_or_predict == 'replay_origin':
            file_path = f"/home/cx/ch_collect_raw_rlbench/{args.task_name}/variation0/episodes/episode{i}/low_dim_obs.pkl"
            with open(file_path, "rb") as f:
                demo = pickle.load(f)
            episode_length = len(demo)
        
        logger.info(f'episode: {i}, steps: {episode_length}')
        obs_dict = env.reset()
        terminated = False
        success = False
        slow_cnt = 0
        cur_robot_state = np.array([ 0.27849028, -0.00815899,  1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ]) if args.use_robot_state else None

        for j in range(episode_length):
            
            # #--------- for key frames replay
            if args.replay_or_predict == 'replay_key':
                action = dat[j]['action']
                robo_state = dat[j]['state']
                action[:3] += robo_state[7:10]
                gripper_open = action[-1]
                action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                action = np.append(action, gripper_open)
                print(j, "  :", action)
                obs_dict, reward, terminated, truncated, info = env.step(action)
                success = success or bool(reward)

            # #----------- for all frames replay
            if args.replay_or_predict == 'replay_origin':
                action = demo[j].gripper_pose
                action = np.append(action, np.array(demo[j].gripper_open))
                print(j, "  :", action)
                obs_dict, reward, terminated, truncated, info = env.step(action)
                success = success or bool(reward)

            # # #----------- for model predict
            if args.replay_or_predict == 'predict':
                image = obs_dict['image']
                image = Image.fromarray(image)
                prompt = env.text

                if slow_cnt % int(args.slow_fast_ratio) == 0:
                    slow_image = image
                    input_ids, slow_latent_embedding = None, None
                    if predict_mode=='diff':
                        input_ids, slow_latent_embedding = model_predict_slow_latent_embedding(model, prompt, slow_image)

                if args.load_pointcloud and args.pointcloud_pos == 'slow' and slow_cnt % int(args.slow_fast_ratio) == 0:
                    point_cloud = obs_dict['point_cloud'] if args.load_pointcloud else None
                elif args.load_pointcloud and args.pointcloud_pos == 'fast':
                    point_cloud = obs_dict['point_cloud'] if args.load_pointcloud else None
                else:
                    point_cloud=None

                output = model_predict(args, predict_mode, model, image, prompt, cur_robot_state, slow_image, point_cloud, input_ids, slow_latent_embedding)

                if predict_mode=='diff':
                    action_diff = output
                    actions = action_diff
                elif predict_mode=='ar':
                    action_ar, predicted_language_subgoals = output
                    actions = action_ar
                    print(predicted_language_subgoals)
                elif predict_mode=='diff+ar':
                    action_diff, action_ar, predicted_language_subgoals = output
                    actions = action_diff
                    print(predicted_language_subgoals)
                
                # Handle dual-arm: if single-arm action, duplicate for dual-arm coordination
                if use_dual_arm and len(actions.shape) > 0:
                    if actions.shape[-1] == 7:
                        # Duplicate single-arm action for both arms
                        actions = np.concatenate([actions, actions], axis=-1)
                        Logger.log_info("Duplicated single-arm action for dual-arm execution")

                for action in actions:
                    robot_state = obs_dict['robot_state']
                    
                    if use_dual_arm:
                        # Dual-arm action: 14 dimensions
                        # [0:7]: UR5e (Robotiq), [7:14]: UR10e (Barrett)
                        if len(action) != 14:
                            # If model outputs single-arm action, duplicate for dual-arm
                            Logger.log_warning("Single-arm action received, duplicating for dual-arm")
                            action = np.concatenate([action, action])
                        
                        # Apply relative positioning for both arms
                        # Left arm (UR5e)
                        action[:3] += robot_state[7:10] if len(robot_state) > 10 else robot_state[:3]
                        # Right arm (UR10e) - use different base position if available
                        if len(robot_state) > 14:
                            action[7:10] += robot_state[14:17]
                        else:
                            action[7:10] += robot_state[7:10]  # Use same base if not available
                        
                        cur_robot_state = action if args.use_robot_state else None
                        
                        # Convert both arms to 7DoF format
                        action_left = EEpose.pose_6DoF_to_7DoF(action[:6])
                        action_left = np.append(action_left, action[6])  # Add gripper
                        action_right = EEpose.pose_6DoF_to_7DoF(action[7:13])
                        action_right = np.append(action_right, action[13])  # Add gripper
                        
                        # Combine for dual-arm action
                        dual_action = np.concatenate([action_left, action_right])
                        logger.info("%d  : Dual-arm action (UR5e+UR10e): %s", j, dual_action)
                        obs_dict, reward, terminated, truncated, info = env.step(dual_action)
                    else:
                        # Single-arm action: 7 dimensions
                        action[:3] += robot_state[7:10]
                        cur_robot_state = action if args.use_robot_state else None
                        gripper_open = action[-1]

                        if args.angle_delta:
                            state_tmp = EEpose.pose_7DoF_to_6DoF(robot_state[7:14])
                            action[3:6] += state_tmp[3:6]
                            
                        action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                        action = np.append(action, gripper_open)
                        logger.info("%d  : %s", j, action)
                        obs_dict, reward, terminated, truncated, info = env.step(action)
                    
                    success = success or bool(reward)
                    if args.sparse and (terminated or truncated or success):
                        break

                slow_cnt += 1
                if terminated or truncated or success:
                    break
                
        if success:
            success_num += 1

        image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "images", f"episode{i}"
        )
        recreate_directory(image_dir)

        env.save_video(os.path.join(video_dir, f'episode{i}_video_steps.mp4'))
        env.save_images(image_dir, quiet=True)
        logger.info(f'episode{i}_{success}')
        Logger.print_seperator()
    
    logger.info(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')
    with open(os.path.join(args.result_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-data-dir', type=str, default='')
    parser.add_argument('--task-name', type=str, default='close_box')
    parser.add_argument('--replay-or-predict', type=str, default='predict')
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--model-action-steps', type=str, default='0')
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--ddim-steps', type=int, default=10)
    parser.add_argument('--llm_middle_layer', type=int, default=32)
    parser.add_argument('--cfg-scale', type=str, default='0')
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--use-diff', type=int, default=1)
    parser.add_argument('--use-ar', type=int, default=1)
    parser.add_argument('--threshold', type=str, default='5.8')
    parser.add_argument('--hf-token', type=str, default='')
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--slow-fast-ratio', type=int, default=4)
    parser.add_argument('--training_mode', type=str, default='async')
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--pointcloud-pos', type=str, default='slow')
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--training-diffusion-steps', type=int, default=100)
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--angle_delta', type=int, default=0)
    parser.add_argument('--lang_subgoals_exist', type=int, default=0)
    parser.add_argument('--use_dual_arm', '--use-dual-arm', type=int, default=0, 
                        dest='use_dual_arm',
                        help='Enable dual-arm mode: UR5e (Robotiq) + UR10e (Barrett BH282) for GF-VLA')
    main(parser.parse_args())