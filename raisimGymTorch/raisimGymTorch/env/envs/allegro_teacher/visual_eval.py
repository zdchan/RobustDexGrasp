#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_teacher as hand
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.allegro_teacher import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats
from scipy.spatial.transform import Rotation as R
from random import choice
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
from datetime import datetime
import argparse
from raisimGymTorch.helper import rotations
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5
import torch


# ===== Configuration Parameters =====
exp_name = "teacher"


# Selected model weights for evaluation
weight_saved = 'teacher_ckpt/full_12500_r.pt'


# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_iterations', type=int, default=50001)
parser.add_argument('-nr', '--num_repeats', type=int, default=1)
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-ln', '--log_name', type=str, default='single_obj')

args = parser.parse_args()
weight_path = args.weight
cfg_grasp = args.cfg

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

# ===== Path and Configuration Setup =====
# Task specification
task_name = args.exp_name
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Directory setup
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# Load configuration from YAML file
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

# Update seed configuration if provided in command line
if args.seed != 1:
    cfg['seed'] = args.seed

# Set number of environments based on command line arguments
num_envs = args.num_repeats
# Set activation function for neural networks
activations = nn.LeakyReLU

# Enable visualization in configuration
cfg['environment']['visualize'] = True
# Update number of environments in configuration
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

# Set dataset for training objects
cat_name = 'new_training_set'
# cat_name = 'shapenet-30obj'

# Whether should load stable states
if cat_name == 'shapenet-30obj':
    stable = True
else:
    stable = False

# Update configuration with dataset information
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)
# Configure single-threaded execution
cfg['environment']['num_threads'] = 1

# Get list of items in the directory
items = os.listdir(directory_path)

# Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

# Initialize list for object paths
obj_path_list = []
# Set the list of available objects
obj_ori_list = folder_names

# Randomly select one object from the list for evaluation
obj_item = choice(obj_ori_list)
# Alternative object options (commented out)
# obj_item = '002_master_chef_can'
# obj_item = '003_cracker_box'
# obj_item = '004_sugar_box'
# obj_item = '005_tomato_soup_can'
# obj_item = '006_mustard_bottle'
# obj_item = '007_tuna_fish_can'
# obj_item = '008_pudding_box'
# obj_item = '009_gelatin_box'
# obj_item = '010_potted_meat_can'
# obj_item = '011_banana'
# obj_item = '019_pitcher_base'
# obj_item = '021_bleach_cleanser'
# obj_item = '024_bowl'
# obj_item = '025_mug'
# obj_item = '035_power_drill'
# obj_item = '036_wood_block'
# obj_item = '037_scissors'
# obj_item = '040_large_marker'
# obj_item = '051_large_clamp'
# obj_item = '052_extra_large_clamp'
# obj_item = '061_foam_brick'
# obj_item = 'blue_pitcher'
# obj_item = 'brush_functional'
# obj_item = 'car_down'
# obj_item = 'fan_small_head'
# obj_item = 'gun_functional'
# obj_item = 'hammer'
# obj_item = 'loopy_head_side'
# obj_item = 'mouse'
# obj_item = 'off_water_body'
# obj_item = 'solder_iron_head'
# obj_item = 'wrench'
# obj_item = 'big_tape'
# obj_item = 'small_tape'
# obj_item = 'small_block'
# obj_item = 'wood_block_oriented'
# obj_item = 'suger_box_oriented'
# obj_item = 'cracker_box_oriented'

# obj_item = 'Bear_34'           
# obj_item = 'Black_mug'      
# obj_item = 'Blue_camera'
# obj_item = 'Blue_teapot'
# obj_item = 'Camera_brown'
# obj_item = 'Camera_yellow'
# obj_item = 'CellPhone_4e'
# obj_item = 'Donut'
# obj_item = 'DrinkBottle_blue_1ef'
# obj_item = 'Gun'
# obj_item = 'Hammer_40'
# obj_item = 'Knife'
# obj_item = 'Mug_8b_red'
# obj_item = 'Mug_gray_d7'
# obj_item = 'Mug_yellow_18'
# obj_item = 'Pan_gray'
# obj_item = 'Plate_gold_69'
# obj_item = 'Purse_brown_d58'
# obj_item = 'Purse_red_f58'
# obj_item = 'Red_bottle'
# obj_item = 'Red_chair'
# obj_item = 'Red_scissor'
# obj_item = 'Stapler_gray_d9'
# obj_item = 'Teapot_blue_high_3d2'
# obj_item = 'Teapot_brown'
# obj_item = 'Vase_red_a1'
# obj_item = 'Watch_9f'
# obj_item = 'Wine_glass2_1_blue'
# obj_item = 'Wine_glass_body_blue'
# obj_item = 'WineGlass_gray_9d'

print(obj_item)

# Initialize vectorized environment with the selected object
env = VecEnv([obj_item], hand.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

# Add object URDF path to the list for loading
obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
# Load articulated objects into the environment
env.load_multi_articulated(obj_path_list)

# Define observation and action dimensions
ob_dim_r = 153  # Observation dimension
act_dim = 22    # Action dimension (joint controls)
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# ===== Training Parameters =====
grasp_steps = cfg['environment']['grasp_steps']  # Number of steps for grasping phase
lift_steps = 30  # Number of steps for lifting phase
n_steps_r = grasp_steps + lift_steps  # Total steps per episode
total_steps_r = n_steps_r * env.num_envs  # Total steps across all environments

# ===== Neural Network Setup =====
# Define actor network with MLP architecture and Gaussian policy
actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

# Define critic network for value function estimation
critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

# Flag for testing directory
test_dir = True

# Setup configuration saver for logging
saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[], test_dir=test_dir)

# Initialize PPO algorithm with actor and critic networks
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
                )

# Load pre-trained policy from specified weight path
load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

# Load lowest points and stable states of objects for proper placement
lowest_points = np.zeros((num_envs, 1), dtype='float32')
if stable:
    stable_states = np.zeros((num_envs, 7), dtype='float32')   
for i in range(num_envs):
    # Read lowest point from file (needed for proper object placement)
    txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())
    if stable:
        stable_state_path = home_path + f"/rsc/{cat_name}/{obj_item}/{obj_item}.npy"
        stable_states[i] = np.load(stable_state_path)[-1, :7]

# ===== Main Evaluation Loop =====
for update in range(args.num_iterations):
    # Start timing for performance measurement
    start = time.time()

    # Initialize reset positions for robot joints and object
    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')  # Right hand joint positions
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')  # Left hand joint positions (not used)
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')  # Object pose: position (3) + quaternion (4) + type (1)

    # Initialize target center for affordance
    target_center = np.zeros_like(env.affordance_center)

    # Set initial finger pose from configuration
    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']

    # Set camera position for hand-centric view
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

    # Initial joint angles for UR5 robot arm
    theta0 = [0.0, -1.57, 1.57, 0., 1.57, -1.57]
    # Joint weights for inverse kinematics
    joint_weights = [1, 1, 1, 1, 1, 1]

    # Initialize inverse kinematics solver for UR5
    ik = InverseKinematicsUR5()
    ik.setJointWeights(joint_weights)
    ik.setJointLimits(-3.14, 3.14)

    # Number of samples for initial pose
    sample_num = cfg['environment']['sample_num']

    # Initialize arrays for visible points
    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')      # Visible points in world frame
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')    # Visible points in object frame

    # Set camera viewpoint for perception
    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    # Initialize environments with randomized object poses
    for i in range(num_envs):
        # Sample object position with random angle and distance
        while True:
            angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
            distance = np.random.uniform(0.45, 0.75)
            sample_x = distance * np.cos(angle)
            sample_y = distance * np.sin(angle)
            if sample_x < 0.25 and sample_x > -0.25:
                break

        # Set object position
        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y
        
        if stable:
            obj_pose_reset[i, 2:7] = stable_states[i, 2:7]
            obj_pose_reset[i, 2] += 0.005
            quats = stable_states[i, 3:7]
        else:
            # Set z-position based on lowest point and use random rotation
            obj_pose_reset[i, 2] = 0.773 - lowest_points[i]  # Adjust z-position based on object's lowest point
            obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]   # Default orientation

            # Generate random rotation around z-axis
            axis_angles = np.zeros((1, 3))
            axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
            quats = rotations.axisangle2quat(axis_angles)
            obj_pose_reset[i, 3:7] = quats  # Set object orientation as quaternion

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

        # Check if top grasp is enabled in configuration
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

        # Sample rotation matrices and calculate projection lengths for hand orientation
        rot_mats, projection_lengths = sample_rot_mats(hand_dir_x_w, sample_num, visible_points_w[i])

        # Initialize arrays to store IK results and feasibility flags
        feasible_ik_flag = np.zeros((sample_num), dtype='bool')
        ik_results = np.zeros((sample_num, 6), dtype='float32')
        
        # Try different hand orientations and find feasible IK solutions
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
                # Valid IK solution found, set joint positions
                qpos_reset_r[i, :6] = ik_result

                # Check for self-collisions by resetting the environment state
                env.reset_state(qpos_reset_r,
                                qpos_reset_l,
                                np.zeros((num_envs, 22), 'float32'),
                                np.zeros((num_envs, 22), 'float32'),
                                obj_pose_reset,
                                )
                # Execute a zero action to update the state
                temp_action_r = np.zeros((num_envs, act_dim), dtype='float32')
                temp_action_l = np.zeros((num_envs, act_dim), dtype='float32')
                _, _, _ = env.step(temp_action_r, temp_action_l)
                
                # Check collision flags in the global state
                global_state = env.get_global_state()
                one_check = global_state[:, 124:128]
                contains_one = np.any(one_check == 1, axis=1)
                true_indices = np.where(contains_one)[0]
                if len(true_indices) > 0:
                    # Collision detected, mark as infeasible
                    feasible_ik_flag[j] = False
                    continue
                else:
                    # No collision, mark as feasible and store IK solution
                    feasible_ik_flag[j] = True
                    ik_results[j, :] = ik_result
                    
        # Find indices of feasible IK solutions
        feasible_indices = np.where(feasible_ik_flag)[0]
        
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

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()
    show_point = dis_info[:, 17:68].astype('float32').copy()
    env.set_joint_sensor_visual(show_point)
    env.update_target(target_center)

    final_actions = np.zeros((num_envs, act_dim), dtype='float32')

    biased = cfg['environment']['biased']
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')

    # ===== Main Action Execution Loop =====
    for step in range(n_steps_r):
        # Time the start of each frame for real-time control
        frame_start = time.time()
        
        # Get current observations
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')

        # Generate action using the actor network
        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        # Initialize left hand actions as zeros (not used)
        action_l = np.zeros_like(action_r)

        # Control logic: grasp phase then lift phase
        if step < grasp_steps:
            # During grasp phase, use actions from the policy network
            final_actions = action_r
        else:
            # During lift phase, use the last grasp action but override arm joints
            action_r = final_actions
            action_r[:, :6] = theta0  # Set arm joints to initial configuration for lifting
            if step == grasp_steps:
                # The transition to lift phase
                print("lift")
                env.switch_root_guidance(True)

        # Execute action in the environment
        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        # Get new observations and sensor data
        obs_new_r, dis_info = env.observe_vision_new()
        show_point = dis_info[:, 17:68].astype('float32').copy()
        env.set_joint_sensor_visual(show_point)

        # Handle biased object positions (simulating uncertainty/disturbances)
        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                # If finger-object distance is small and object not yet biased
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                    # Mark object as biased and apply position bias
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            # Update object positions with bias
            env.switch_obj_pos(obj_pos_bias_current)

        # Maintain real-time control frequency
        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    print("end")

# ===== End of Evaluation =====
# This script performs visualized robotic grasping evaluation using a pre-trained policy.
# It loads a specified model checkpoint and evaluates the grasping performance on objects.
# The evaluation process includes:
# 1. Sampling and positioning objects in the environment
# 2. Extracting point cloud from objects
# 3. Calculating optimal hand approach directions and orientations
# 4. Solving inverse kinematics for feasible grasp poses
# 5. Executing the grasp-then-lift sequence using the trained policy
# 6. Optionally applying position biases to test robustness
#
# The results can be visualized in the RaiSim environment if visualization is enabled.



