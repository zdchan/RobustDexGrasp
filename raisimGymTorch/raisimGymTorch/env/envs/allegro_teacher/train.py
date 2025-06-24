#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_teacher as hand
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.allegro_teacher import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats

# Enable line buffering for real-time output printing
import sys
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
from raisimGymTorch.helper import rotations
import random
import wandb
import torch

from random import choices
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter

# ===== Configuration Parameters =====
# Set experiment name
exp_name = "teacher"

# Path to pre-trained model for continued training (if enabled)
weight_saved = '/../teacher_ckpt/full_12500_r.pt'

# ===== Command Line Argument Parsing =====
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_iterations', type=int, default=50001)
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-ln', '--log_name', type=str, default=None)

# Parse the arguments
args = parser.parse_args()
weight_path = args.weight
cfg_grasp = args.cfg

# Print configuration information
print(f"Configuration file: \"{args.cfg}\"", file=sys.stdout)
print(f"Experiment name: \"{args.exp_name}\"", file=sys.stdout)

# ===== Path and Configuration Setup =====
# Task specification
task_name = args.exp_name
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Directory setup
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# Set experiment path based on arguments
if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# Load configuration from YAML file
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

# ===== Initialize Experiment Tracking =====
# Initialize Weights & Biases if log_name is provided
if args.log_name is not None:
    wandb.init(project=task_name, config=cfg, name=args.log_name)

# Set random seed if specified
if args.seed != 1:
    cfg['seed'] = args.seed

# ===== Object Loading Setup =====
obj_path_list = []
obj_list = []

# Set dataset type for training
cat_name = 'new_training_set'

# Set number of repetitions per object
repeat_per_obj = 2

# Update configuration with dataset information
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path, file=sys.stdout)
# Get list of items in the dataset directory
items = os.listdir(directory_path)

# Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

# Initialize object lists
obj_path_list = []
obj_ori_list = folder_names

# Increase the number of repetitions for difficult objects to improve training
obj_ori_list.append('037_scissors')
obj_ori_list.append('037_scissors')
obj_ori_list.append('off_water_body')
obj_ori_list.append('off_water_body')
obj_ori_list.append('019_pitcher_base')
obj_ori_list.append('011_banana')
obj_ori_list.append('mouse')
obj_ori_list.append('hammer')
obj_ori_list.append('small_block')

# Calculate total number of environments based on objects and repetitions
num_envs = len(obj_ori_list) * repeat_per_obj
# Create the complete object list with repetitions
for i in range(repeat_per_obj):
    for item in obj_ori_list:
        obj_list.append(item)

# Set activation function for neural networks
activations = nn.LeakyReLU

# Configure visualization mode when running without logging (for debugging)
if args.log_name is None:
    # For local testing without logging, use fewer environments and enable visualization
    num_envs = repeat_per_obj
    obj_list = choices(obj_list, k=1)  # Select just one object
    obj_list.append(obj_list[0])       # Duplicate it for multiple trials
    obj_list.append(obj_list[0])
    cfg['environment']['visualize'] = True

# ===== Sampling Configuration =====
# Check if non-uniform sampling flag exists in configuration, default to False if not
non_uniform_sampling = cfg['environment'].get('non_uniform_sampling', False)
if non_uniform_sampling:
    print("Training with non-uniform sampling (biased towards edges)", file=sys.stdout)
else:
    print("Training with uniform sampling", file=sys.stdout)
print("Evaluation will always use uniform sampling for fair assessment", file=sys.stdout)

# Update environment configuration with number of environments
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs, file=sys.stdout)

# ===== Environment Setup =====
# Create vectorized environment with specified objects
env = VecEnv(obj_list, hand.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

# Load object models into environment
for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)

# ===== Model Dimension Setup =====
# Define observation and action dimensions
ob_dim_r = 153  # Observation dimension
act_dim = 22    # Action dimension (joint controls)
print('ob dim', ob_dim_r, file=sys.stdout)
print('act dim', act_dim, file=sys.stdout)

# ===== Training Parameters =====
# Configure reward clipping to prevent extreme updates
reward_clip = -2.0
# Number of steps per grasp episode
n_steps_r = cfg['environment']['grasp_steps']
# Total steps across all environments
total_steps_r = n_steps_r * env.num_envs

# ===== Build Neural Network Models =====
# Actor network
actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

# Critic network
critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

# Flag for testing directory, set to False for normal training
test_dir = False

# ===== Setup Configuration Saver =====
# Configure saver for model checkpoints and logging
saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/train.py", task_path + "/../../RaisimGymVecEnvOther.py"], test_dir=test_dir)

# ===== Initialize PPO Algorithm =====
ppo_r = PPO.PPO(actor=actor_r,
                critic=critic_r,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False
                # learning_rate=1e-4
                )

# ===== Load Pre-trained Student Model (if specified) =====
if args.load_trained_policy:
    load_param(saver.data_dir.split('eval')[0] + weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir,
               cfg_grasp)

# ===== Initialize Training Variables =====
# Set finger weights for reward calculation (adjust importance of different fingers)
finger_weights = np.ones((num_envs, 17)).astype('float32')
# Increase weight of thumb and fingertips for better grasp
for i in range(4):
    finger_weights[:, 4 * i+4] *= 4.0  # Increase weight for fingertips
finger_weights[:, 16] *= 2.0           # Increase weight for thumb
# Normalize weights to sum to 1, then scale up
finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
finger_weights[:, 0] = 0.0             # Zero weight for palm contact
finger_weights *= 16.0                 # Scale up for stronger gradient signal

# Initialize reward components
affordance_reward_r = np.zeros((num_envs, 1))
table_reward_r = np.zeros((num_envs, 1))
arm_height_reward_r = np.zeros((num_envs, 1))
arm_collision_reward_r = np.zeros((num_envs, 1))


# Initialize state variables for robot and objects
qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')  # Right hand joint positions
qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')  # Left hand joint positions (not used)
obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')  # Object poses [position(3), quaternion(4), type(1)]

# Index of the most recently saved checkpoint
saved_update_idx = 0

# ===== Load Object Data =====
# Load lowest points of objects for proper placement
lowest_points = np.zeros((num_envs, 1), dtype='float32')
stable_states = np.zeros((num_envs, 7), dtype='float32')
for i in range(num_envs):
    # Read lowest point data for each object from text file
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())


# ===== Main Training Loop =====
for update in range(args.num_iterations):
    # Start timing for performance measurement
    start = time.time()

    # ===== Evaluation Mode Check =====
    # Determine if current iteration should run evaluation
    if cfg['environment']['eval_during_training']:
        # Run evaluation every 100 iterations
        is_evaluation = update % 100 == 0 
    else:
        is_evaluation = False
    
    if is_evaluation:
        print(f"Evaluating policy at iteration {update}...", file=sys.stdout)
        print("Using uniform sampling for fair evaluation", file=sys.stdout)
        # In evaluation mode, set lift_steps to 100 to test lifting capability
        lift_steps = 100
        current_steps = n_steps_r + lift_steps
    else:
        # Normal training mode (no lifting phase)
        lift_steps = 0
        current_steps = n_steps_r
    
    # Update total steps for current iteration
    total_steps = current_steps * env.num_envs

    
    # ===== Checkpoint Saving =====
    # Save model and visualize performance periodically
    if update % cfg['environment']['eval_every_n'] == 0 and args.log_name is not None:
        print("Visualizing and evaluating the current policy", file=sys.stdout)
        # Save full model state including actor, critic, and optimizer
        torch.save({
            'actor_architecture_state_dict': actor_r.architecture.state_dict(),
            'actor_distribution_state_dict': actor_r.distribution.state_dict(),
            'critic_architecture_state_dict': critic_r.architecture.state_dict(),
            'optimizer_state_dict': ppo_r.optimizer.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_r.pt')

        # Save environment scaling parameters
        env.save_scaling(saver.data_dir, str(update))
        saved_update_idx = update

    # Initialize target center for affordance (visual target)
    target_center = np.zeros_like(env.affordance_center)

    # Initialize finger pose from configuration
    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']


    # ===== Point Cloud and Vision Setup =====
    # Initialize arrays for visible points
    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')    # Visible points in world frame
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')  # Visible points in object frame

    # Set camera viewpoint for perception
    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]
    
    # Set hand direction starting point (camera position used as reference)
    heading_dir_start_w = np.zeros((1, 3))
    heading_dir_start_w[0, 0] = cfg['environment']['camera_position'][0]
    heading_dir_start_w[0, 1] = cfg['environment']['camera_position'][1]
    heading_dir_start_w[0, 2] = cfg['environment']['camera_position'][2]

    # Set wrist bias (offset for proper hand positioning, from hand center to wrist for IK calculation)
    hand_center = np.zeros((1, 3))
    hand_center[0, 0] = cfg['environment']['hardware']['hand_center'][0]
    hand_center[0, 1] = cfg['environment']['hardware']['hand_center'][1]
    hand_center[0, 2] = cfg['environment']['hardware']['hand_center'][2]

    # Transformation matrix from UR5 robot frame to world frame
    ur5_to_world = np.eye(3)
    ur5_to_world[0, 0] = 0
    ur5_to_world[0, 1] = -1
    ur5_to_world[1, 0] = 1
    ur5_to_world[1, 1] = 0

    # Initial joint angles and weights for IK solver
    theta0 = [0.0, -1.57, 1.57, 0., 1.57, -1.57]
    joint_weights = [1, 1, 1, 1, 1, 1]

    # Initialize inverse kinematics solver for UR5 robot arm
    ik = InverseKinematicsUR5()
    ik.setJointWeights(joint_weights)
    ik.setJointLimits(-3.14, 3.14)

    # Number of IK samples to try for each object
    sample_num = cfg['environment']['sample_num']


    # ===== Process Each Environment =====
    for i in range(num_envs):
        # ===== Object Initialization =====
        # Randomly sample object position
        if is_evaluation or not non_uniform_sampling:
            # For evaluation or when uniform sampling is enabled
            # Use uniform sampling within a constrained region
            while True:
                angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
                distance = np.random.uniform(0.45, 0.75)
                sample_x = distance * np.cos(angle)
                sample_y = distance * np.sin(angle)
                if sample_x < 0.25 and sample_x > -0.25:
                    break
        else:        
            if np.random.random() < 0.5:
                # 50% chance of using uniform sampling
                while True:
                    angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
                    distance = np.random.uniform(0.45, 0.75)
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_x < 0.25 and sample_x > -0.25:
                        break
            else:
                # 50% chance of using edge-biased sampling for curriculum learning
                # Beta distribution creates a U-shaped distribution with higher probability at edges
                while True:
                    # For angle, use Beta distribution to make edge probabilities higher
                    # Beta(0.5, 0.5) is U-shaped, with higher probabilities near 0 and 1
                    beta_param = 0.5
                    angle_normalized = np.random.beta(beta_param, beta_param)  # In range [0,1], higher probability at ends
                    angle = -0.7 * np.pi + angle_normalized * (0.4 * np.pi) # Map to [-0.7π, -0.3π] range
                    # For distance, also use Beta distribution to make endpoints more likely
                    distance_normalized = np.random.beta(beta_param, beta_param)  # In range [0,1], higher probability at ends
                    distance = 0.45 + distance_normalized * 0.3  # Map to [0.45, 0.75]
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_x < 0.25 and sample_x > -0.25:
                        break

        # Set object position
        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y

        # Set object z-position based on lowest point
        obj_pose_reset[i, 2] = 0.773 - lowest_points[i]  # Adjust height based on object's lowest point
        # Initialize with default orientation
        obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

        # Generate random rotation around z-axis
        axis_angles = np.zeros((1, 3))
        axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
        quats = rotations.axisangle2quat(axis_angles)
        obj_pose_reset[i, 3:7] = quats  # Set object orientation as quaternion

        # ===== Point Cloud Processing =====
        # Convert quaternion to rotation matrix for object
        obj_mat_single = rotations.quat2mat(quats).reshape(3, 3)

        # Calculate ray origins and directions for visibility testing
        view_point_obj_diff = view_point_world - obj_pose_reset[i, :3]
        view_point_obj = np.matmul(obj_mat_single.T, view_point_obj_diff.T).T
        obj_pcd = env.affordance_pcd[i].reshape(200, 3).cpu().numpy()
        directions = obj_pcd - view_point_obj
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        
        # Perform ray-tracing to determine visible points on the object
        locations, index_ray, index_tri = env.aff_mesh[i].ray.intersects_location(ray_origins=view_point_obj,
                                                                                    ray_directions=directions,
                                                                                    multiple_hits=False)
        # Ensure consistent dimensionality of locations array
        if locations.shape != (200, 3):
            expanded_locations = np.zeros((200, 3))
            expanded_locations[:, :] = locations[0, :]
            expanded_locations[:locations.shape[0], :] = locations
            locations = expanded_locations
            
        # Store visible points in both object and world coordinates
        visible_points_obj[i, :] = locations
        visible_points_w[i, :] = np.matmul(obj_mat_single, locations.T).T + obj_pose_reset[i, :3]

        # Calculate center of visible points (affordance center) in world frame
        obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200, 3), axis=0)

        # ===== Hand Approach Direction =====
        top_grasp = cfg['environment']['top']

        # Determine approach direction for hand
        if top_grasp:
            # For top grasp, approach from above (z-direction)
            hand_dir_x_w = np.zeros((1, 3))
            hand_dir_x_w[0, 2] = 1
        else:
            # For side grasp, approach from camera direction
            hand_dir_x_w = heading_dir_start_w - obj_aff_center_in_w
            hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)

        # Calculate wrist position based on affordance center and approach direction
        pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w

        # ===== Inverse Kinematics Solving =====
        # Sample rotation matrices and calculate projection lengths for hand orientation
        rot_mats, projection_lengths = sample_rot_mats(hand_dir_x_w, sample_num, visible_points_w[i])

        # Check if IK is feasible for each orientation
        feasible_ik_flag = np.zeros((sample_num), dtype='bool')
        ik_results = np.zeros((sample_num, 6), dtype='float32')
        
        for j in range(sample_num):
            rot_mat = rot_mats[j]
            wrist_in_world = rot_mat
            # Apply wrist bias in world frame
            hand_center_in_world = np.matmul(wrist_in_world, hand_center.T).T

            # Calculate position in UR5 robot frame
            pos_in_ur5 = np.zeros((3, 1))
            pos_in_ur5[0, 0] = pos[0, 0] - 0. + hand_center_in_world[0, 0]
            pos_in_ur5[1, 0] = pos[0, 1] - 0. + hand_center_in_world[0, 1]
            pos_in_ur5[2, 0] = pos[0, 2] - 0.771 + hand_center_in_world[0, 2]
            pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)

            # Transform wrist orientation to UR5 frame
            wrist_mat_in_ur5 = np.matmul(ur5_to_world.T, wrist_in_world)

            # Create goal transformation matrix for IK solver
            gd = np.eye(4)
            gd[:3, :3] = wrist_mat_in_ur5
            gd[0, 3] = pos_in_ur5_new[0, 0]
            gd[1, 3] = pos_in_ur5_new[1, 0]
            gd[2, 3] = pos_in_ur5_new[2, 0]

            # Solve inverse kinematics to find joint angles
            ik_result = ik.findClosestIK(gd, theta0)
            if ik_result is None or np.isnan(ik_result).any():
                # IK solution not found or contains NaN values
                feasible_ik_flag[j] = False
                continue
            else:
                # Valid IK solution found, store the result
                ik_results[j, :] = ik_result
                feasible_ik_flag[j] = True

        # Find indices of feasible IK solutions
        feasible_indices = np.where(feasible_ik_flag)[0]
        
        # ===== Select Best IK Solution =====
        # Initialize scores for ranking different solutions
        scores = np.ones(sample_num, dtype='float32')
        scores = scores * 10000.  # Set initial scores high (worse)
        
        # Select the best IK solution based on projection length and joint angles
        if min(projection_lengths) < 0.18:
            # If we have short projection lengths, use scoring based on multiple criteria
            for j in feasible_indices:
                if projection_lengths[j] < 0.18:
                    # Calculate score based on projection length and wrist angle
                    score1 = projection_lengths[j] * cfg['environment']['length_score_coeff']
                    score2 = abs(ik_results[j, 4] - 1.57) * cfg['environment']['angle_score_coeff']
                    score3 = (abs(ik_results[j, 4]) - 3.2) * cfg['environment']['angle_score_coeff'] * 0.5
                    scores[j] = score1 + score2 + score3
                else:
                    scores[j] = 10000.  # Keep high score for solutions with long projections
            best_index = np.argmin(scores)
            qpos_reset_r[i, :6] = ik_results[best_index]
        else:
            # If all projection lengths are large, simply use the shortest one
            best_index = np.argmin(projection_lengths)
            qpos_reset_r[i, :6] = ik_results[best_index]

    # check self collision
    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )
    temp_action_r = np.zeros((num_envs, act_dim), dtype='float32')
    temp_action_l = np.zeros((num_envs, act_dim), dtype='float32')
    _, _, _ = env.step(temp_action_r, temp_action_l)
    global_state = env.get_global_state()
    one_check = global_state[:, 124:128]
    contains_one = np.any(one_check == 1, axis=1)
    true_indices = np.where(contains_one)[0]
    for true_idx in true_indices:
        current_obj_idx = true_idx // repeat_per_obj
        current_obj_env_indices = []
        for i in range(repeat_per_obj):
            current_obj_env_indices.append(current_obj_idx * repeat_per_obj + i)
        false_indices = [idx for idx in current_obj_env_indices if not contains_one[idx]]
        if len(false_indices)>0:
            chosen_index = np.random.choice(false_indices)
            qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
            obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
        else:
            qpos_reset_r[true_idx, :6] = [angle+np.pi/2-0.3, -1.57, 1.57, 0., 1.57, -1.57]
            obj_pose_reset[true_idx, 0] = 0.1
            obj_pose_reset[true_idx, 1] = -0.5


    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()
    env.update_target(target_center)
    rewards_r_sum = env.get_reward_info_r()
    for i in range(len(rewards_r_sum)):
        rewards_r_sum[i]['affordance_reward'] = 0
        rewards_r_sum[i]['table_reward'] = 0
        rewards_r_sum[i]['arm_height_reward'] = 0
        rewards_r_sum[i]['arm_collision_reward'] = 0

        for k in rewards_r_sum[i].keys():
            rewards_r_sum[i][k] = 0


    biased = cfg['environment']['biased']
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')


    for step in range(current_steps):
        obs_r = obs_new_r
        obs_r = obs_r[:].astype('float32')

        action_r = ppo_r.act(obs_r)
        action_l = np.zeros_like(action_r)

        # If in evaluation mode and grasp phase is completed, enter lift phase
        if is_evaluation and step >= n_steps_r:
            # In lift phase, use fixed arm pose
            action_r[:, :6] = theta0
            if step == n_steps_r:
                # Enable root guidance at the start of lifting
                env.switch_root_guidance(True)
                print("Starting lift phase...", file=sys.stdout)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        obs_new_r = obs_new_r[:].astype('float32')


        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)


        global_state = env.get_global_state()

        rewards_r = env.get_reward_info_r()
        affordance_reward_r = - np.sum((dis_info[:, 1:17]) * finger_weights[:, 1:17], axis=1)
        table_reward_r = -np.sum(np.log(50*np.clip(obs_new_r[:, 70:87], a_min=0.002, a_max=0.02)) * finger_weights, axis=1)
        arm_height_reward_r = -np.sum(np.log(50*np.clip(obs_new_r[:, 89:93], a_min=0.002, a_max=0.02)), axis=1)

        one_check = global_state[:, 124:128]
        arm_collision_reward_r = np.sum(one_check, axis=1)

        for i in range(num_envs):
            rewards_r[i]['affordance_reward'] = affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff']
            rewards_r[i]['table_reward'] = table_reward_r[i] * cfg['environment']['reward']['table_reward']['coeff']
            rewards_r[i]['arm_height_reward'] = arm_height_reward_r[i] * cfg['environment']['reward']['arm_height_reward']['coeff']
            rewards_r[i]['arm_collision_reward'] = arm_collision_reward_r[i] * cfg['environment']['reward']['arm_collision_reward']['coeff']

            rewards_r[i]['reward_sum'] = (
                        rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] +
                        rewards_r[i]['table_reward'] + rewards_r[i]['arm_height_reward'] + 
                        rewards_r[i]['arm_collision_reward'])

            reward_r[i] = rewards_r[i]['reward_sum']
        reward_r.clip(min=reward_clip)

        for i in range(len(rewards_r_sum)):
            for k in rewards_r_sum[i].keys():
                rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]

        # Only collect training data in non-evaluation mode
        if not is_evaluation:
            ppo_r.step(value_obs=obs_r, rews=reward_r, dones=dones)

    # If in evaluation mode, calculate success rate
    if is_evaluation:
        # Get global state to check if objects were lifted
        global_state = env.get_global_state()
        lifted = global_state[:, 107] - obj_pose_reset[:, 2] > 0.1
        success_rate = np.sum(lifted) / num_envs
        
        # Print current success rate
        print(f"Evaluation success rate at iteration {update}: {success_rate:.4f}", file=sys.stdout)
        
        # Calculate and print success rate for each object
        print("\n===== Per-Object Success Rate =====", file=sys.stdout)
        # Create a dictionary to store success and attempt counts for each object
        object_success = {}
        for i in range(num_envs):
            obj_name = obj_list[i]
            if obj_name not in object_success:
                object_success[obj_name] = {"success": 0, "attempts": 0}
            object_success[obj_name]["attempts"] += 1
            if lifted[i]:
                object_success[obj_name]["success"] += 1
        
        # Print object type statistics
        print(f"Total object types: {len(object_success)}", file=sys.stdout)
        print(f"Total environments: {num_envs}", file=sys.stdout)
        
        # Print success rate for each object, sorted from lowest to highest
        sorted_objects = sorted(
            object_success.items(),
            key=lambda x: (x[1]["success"] / x[1]["attempts"]) if x[1]["attempts"] > 0 else 0
        )
        for obj_name, stats in sorted_objects:
            success_rate_obj = (stats["success"] / stats["attempts"]) if stats["attempts"] > 0 else 0
            print(f"{obj_name:<30} Success: {stats['success']}/{stats['attempts']} ({success_rate_obj:.2%})", file=sys.stdout)
        
        print("=====================================\n", file=sys.stdout)
        
        
        # If using wandb, log evaluation results
        if args.log_name is not None:
            wandb.log({"evaluation_success_rate": success_rate}, step=update)
            # Also log success rate for each object
            for obj_name, stats in object_success.items():
                success_rate_obj = (stats["success"] / stats["attempts"]) if stats["attempts"] > 0 else 0
                wandb.log({f"object_success_rate/{obj_name}": success_rate_obj}, step=update)
        
        # Disable root guidance
        env.switch_root_guidance(False)

    obs_r, _ = env.observe_vision_new()
    obs_r = obs_r[:, :].astype('float32')

    if np.isnan(obs_r).any():
        print('nan in obs', file=sys.stdout)
        print(obs_r)

    # Only update policy in non-evaluation mode
    if not is_evaluation:
        # update policy
        ppo_r.update(actor_obs=obs_r, value_obs=obs_r, log_this_iteration=update % 10 == 0, update=update)

    actor_r.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    if ppo_r.check_exploding_gradient():
        print("------------------- exploding gradient !!! will reload param --------------------", file=sys.stdout)
        ppo_r.is_exploding_gradient = False
        load_pth = saver.data_dir + "/full_" + str(saved_update_idx) + '_r.pt'
        load_param(load_pth, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)


    end = time.time()

    ave_reward = {}
    for k in rewards_r_sum[0].keys():
        ave_reward[k] = 0
    for k in rewards_r_sum[0].keys():
        for i in range(len(rewards_r_sum)):
            ave_reward[k] = ave_reward[k] + rewards_r_sum[i][k]
        # When calculating average rewards, only consider grasp phase steps
        ave_reward[k] = ave_reward[k] / (len(rewards_r_sum) * n_steps_r)
    
    # Only log training rewards in non-evaluation mode
    if args.log_name is not None and not is_evaluation:  # 只在非评估阶段记录训练奖励
        wandb.log(ave_reward, step=update)

    if args.log_name is None:
        print(ave_reward, file=sys.stdout)

    print('----------------------------------------------------', file=sys.stdout)
    print('{:>6}th iteration'.format(update), file=sys.stdout)
    print('{:<40} {:>6}'.format("average reward: ", '{:0.10f}'.format(ave_reward['reward_sum'])), file=sys.stdout)
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)), file=sys.stdout)
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))), file=sys.stdout)
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                   * cfg['environment']['control_dt'])), file=sys.stdout)
    # print('std: ')
    # print(np.exp(actor_r.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n', file=sys.stdout)