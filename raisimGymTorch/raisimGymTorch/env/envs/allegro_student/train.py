#!/usr/bin/python
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_student as hand
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
from raisimGymTorch.env.bin.allegro_student import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats
import sys
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
import os
import math
import time
import raisimGymTorch.algo.ppo_dagger_recon.module as ppo_module
from raisimGymTorch.algo.ppo_dagger_recon.dagger_partial import Dagger
import torch.nn as nn
import numpy as np
import torch
import argparse
from raisimGymTorch.helper import rotations
import random
import wandb
import torch

from random import choices
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5

# ===== Configuration Parameters =====
exp_name = "student"

# Path to pre-trained teacher model
weight_teacher = '/../../teacher/teacher_ckpt/full_12500_r.pt'

# Path to student model for continued training (if enabled)
weight_path_student = '/../student_ckpt/full_5500_r.pt'

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_teacher)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_iterations', type=int, default=50001)
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-ln', '--log_name', type=str, default=None)

args = parser.parse_args()
weight_path = args.weight
cfg_grasp = args.cfg

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

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# Load configuration from YAML file
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

# Initialize Weights & Biases if log_name is provided
if args.log_name is not None:
    wandb.init(project=task_name, config=cfg, name=args.log_name)

# Set random seed if specified
if args.seed != 1:
    cfg['seed'] = args.seed

# ===== Object Loading Setup =====
obj_path_list = []
obj_list = []

cat_name = 'new_training_set'

# Number of repetitions per object
repeat_per_obj = 2

# Set loading set
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path, file=sys.stdout)
items = os.listdir(directory_path)

# Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_path_list = []
obj_ori_list = folder_names
# Increase the number of repetitions for difficult objects
obj_ori_list.append('037_scissors')
obj_ori_list.append('037_scissors')
obj_ori_list.append('off_water_body')
obj_ori_list.append('off_water_body')
obj_ori_list.append('019_pitcher_base')
obj_ori_list.append('011_banana')
obj_ori_list.append('mouse')
obj_ori_list.append('hammer')
obj_ori_list.append('small_block')

# Calculate total environments based on objects and repetitions
num_envs = len(obj_ori_list) * repeat_per_obj
obj_list = []
for i in range(repeat_per_obj):
    for item in obj_ori_list:
        obj_list.append(item)

# Set activation function for neural networks
activations = nn.LeakyReLU

# Configure visualization mode when running without logging (for debugging)
if args.log_name is None:
    num_envs = repeat_per_obj
    obj_list = choices(obj_list, k=num_envs)
    cfg['environment']['visualize'] = True


cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs, file=sys.stdout)

# ===== Sampling Configuration =====
# Check if non-uniform sampling flag exists in configuration, default to False if not
non_uniform_sampling = cfg['environment'].get('non_uniform_sampling', False)
if non_uniform_sampling:
    print("Training with non-uniform sampling (biased towards edges)", file=sys.stdout)
else:
    print("Training with uniform sampling", file=sys.stdout)
print("Evaluation will always use uniform sampling for fair assessment", file=sys.stdout)

# ===== Environment Setup =====
# Create vectorized environment with specified objects
env = VecEnv(obj_list, hand.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

# Load object models into environment
for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


# ===== Training Parameters =====
reward_clip = -2.0
n_steps_r = cfg['environment']['grasp_steps']
total_steps_r = n_steps_r * env.num_envs

test_dir = False

# Setup configuration saver for checkpoints and logging
saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/train.py"], test_dir=test_dir)


# ===== Model Dimension Setup =====
ob_dim_r = 153
act_dim = 22
print('ob dim', ob_dim_r, file=sys.stdout)
print('act dim', act_dim, file=sys.stdout)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim=26
aff_vec_dim = 51
total_obs_dim = tobeEncode_dim*t_steps + ob_dim_r

# ===== Training Strategy Configuration =====
update_mlp = True
student_driven_ratio=1.0
if update_mlp:
    ppo_ratio = 0.5
else:
    ppo_ratio = 0

print('update mlp: ', update_mlp, file=sys.stdout)
print('student driven ratio: ', student_driven_ratio, file=sys.stdout)
print('ppo ratio: ', ppo_ratio, file=sys.stdout)

# ===== Build Neural Network Models =====
# Expert actor network
actor_expert_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
# Student actor network
actor_student_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
# Student critic network
critic_student_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


print('loading expert policy from: ', saver.data_dir.split('eval')[0] + weight_path, file=sys.stdout)

# Load expert policy weights
checkpoint = torch.load(saver.data_dir.split('eval')[0] + weight_path, map_location=torch.device(device))
actor_expert_r.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
actor_expert_r.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])

expert_policy = actor_expert_r.architecture
# Initialize property latent encoder (LSTM-based state history encoder)
prop_latent_encoder = ppo_module.LSTM_StateHistoryEncoder(tobeEncode_dim, prop_latent_dim, t_steps, device).to(device)

# ===== Initialize DAgger Algorithm =====
dagger = Dagger(expert_policy=expert_policy,
                actor_student=actor_student_r,
                critic_student=critic_student_r,
                prop_latent_encoder=prop_latent_encoder,
                tobeEncode_dim=tobeEncode_dim,
                prop_latent_dim=prop_latent_dim,
                total_obs_dim=total_obs_dim,
                mlp_obs_dim=ob_dim_r,
                t_steps=t_steps,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False,
                update_mlp=update_mlp,
                ppo_ratio=ppo_ratio
                )

# ===== Load Pre-trained Student Model (if specified) =====
if args.load_trained_policy:
    print('loading trained policy from: ', saver.data_dir.split('eval')[0] + weight_path_student)
    checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device(device))

    actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
    actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
    critic_student_r.architecture.load_state_dict(checkpoint_student['critic_architecture_state_dict'])
    prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

else:
    # If not loading pre-trained student, initialize student with expert weights
    actor_student_r.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor_student_r.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic_student_r.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])


# ===== Initialize Training Variables =====
# Set finger weights for reward calculation (adjust importance of different fingers)
finger_weights = np.ones((num_envs, 17)).astype('float32')
for i in range(4):
    finger_weights[:, 4 * i+4] *= 4.0
finger_weights[:, 16] *= 2.0
finger_weights[:, 0] = 0.0
finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
finger_weights *= 16.0

# Initialize reward components
affordance_reward_r = np.zeros((num_envs, 1))
table_reward_r = np.zeros((num_envs, 1))
arm_height_reward_r = np.zeros((num_envs, 1))
arm_collision_reward_r = np.zeros((num_envs, 1))

# Initialize state variables
qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

# Load object lowest points data (for proper placement)
lowest_points = np.zeros((num_envs, 1), dtype='float32')
stable_states = np.zeros((num_envs, 7), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())

for update in range(args.num_iterations):

    # ===== Curriculum Learning Configuration =====
    if cfg['environment']['curriculum']:
        ppo_ratio = min(update*0.0005, 1.0)
        dagger.update_ppo_ratio(ppo_ratio)
    if update % 1000 == 0:
        print('ppo ratio: ', ppo_ratio, file=sys.stdout)
        print('student driven ratio: ', student_driven_ratio, file=sys.stdout)

    # ===== Evaluation Mode Configuration =====
    if cfg['environment']['eval_during_training']:
        # Run evaluation every 100 iterations
        is_evaluation = update % 100 == 0 
    else:
        is_evaluation = False
    
    if is_evaluation:
        print(f"Evaluating policy at iteration {update}...", file=sys.stdout)
        print("Using uniform sampling for fair evaluation", file=sys.stdout)
        # In evaluation mode, set lift_steps to 100
        lift_steps = 100
        current_steps = n_steps_r + lift_steps
    else:
        # Normal training mode
        lift_steps = 0
        current_steps = n_steps_r
    
    # Update total steps for current iteration
    total_steps = current_steps * env.num_envs

    start = time.time()

    # ===== Checkpoint Saving =====
    # Save model and visualize performance periodically
    if update % cfg['environment']['eval_every_n'] == 0 and args.log_name is not None:
        print("Visualizing and evaluating the current policy", file=sys.stdout)
        # Save model weights
        torch.save({
            'actor_architecture_state_dict': actor_student_r.architecture.state_dict(),
            'actor_distribution_state_dict': actor_student_r.distribution.state_dict(),
            'critic_architecture_state_dict': critic_student_r.architecture.state_dict(),
            'optimizer_state_dict': dagger.optimizer.state_dict(),
            'prop_latent_encoder_state_dict': prop_latent_encoder.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_r.pt')

        # Save environment scaling parameters
        env.save_scaling(saver.data_dir, str(update))

    target_center = np.zeros_like(env.affordance_center)

    # Initialize finger pose from configuration
    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']


    # ===== Point Cloud and Vision Setup =====
    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    # Set camera viewpoint
    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    # Set hand direction starting point
    heading_dir_start_w = np.zeros((1, 3))
    heading_dir_start_w[0, 0] = cfg['environment']['camera_position'][0]
    heading_dir_start_w[0, 1] = cfg['environment']['camera_position'][1]
    heading_dir_start_w[0, 2] = cfg['environment']['camera_position'][2]

    # Set wrist bias (offset for proper hand positioning, from hand center to wrist for IK calculation)
    hand_center = np.zeros((1, 3))
    hand_center[0, 0] = cfg['environment']['hardware']['hand_center'][0]
    hand_center[0, 1] = cfg['environment']['hardware']['hand_center'][1]
    hand_center[0, 2] = cfg['environment']['hardware']['hand_center'][2]
    # UR5 robot to world coordinate system transformation matrix
    ur5_to_world = np.eye(3)
    ur5_to_world[0, 0] = 0
    ur5_to_world[0, 1] = -1
    ur5_to_world[1, 0] = 1
    ur5_to_world[1, 1] = 0

    # Initial joint angles and weights for IK solver
    theta0 = [0.0, -1.57, 1.57, 0., 1.57, -1.57]
    joint_weights = [1, 1, 1, 1, 1, 1]

    # Initialize inverse kinematics solver
    ik = InverseKinematicsUR5()
    ik.setJointWeights(joint_weights)
    ik.setJointLimits(-3.14, 3.14)
    
    # Number of IK samples to try
    sample_num = cfg['environment']['sample_num']

    for i in range(num_envs):
        if is_evaluation or not non_uniform_sampling:
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
                # 50% chance of using edge-biased sampling
                while True:
                    # For angle, use Beta distribution to make edge probabilities higher. Beta(0.5, 0.5) is U-shaped, with higher probabilities near 0 and 1
                    beta_param = 0.5
                    angle_normalized = np.random.beta(beta_param, beta_param)  # In range [0,1], higher probability at ends
                    angle = -0.7 * np.pi + angle_normalized * (0.4 * np.pi) # Map to [-0.7π, -0.3π] range
                    # For distance, also use Beta(0.5, 0.5) distribution to make endpoints more likely
                    distance_normalized = np.random.beta(beta_param, beta_param)  # In range [0,1], higher probability at ends
                    distance = 0.45 + distance_normalized * 0.3  # Map to [0.45, 0.75]
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_x < 0.25 and sample_x > -0.25:
                        break

        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y
        obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
        obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

        axis_angles = np.zeros((1, 3))
        axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
        quats = rotations.axisangle2quat(axis_angles)
        obj_pose_reset[i, 3:7] = quats

        # Get the partial point cloud
        obj_mat_single = rotations.quat2mat(quats).reshape(3, 3)

        view_point_obj_diff = view_point_world - obj_pose_reset[i, :3]
        view_point_obj = np.matmul(obj_mat_single.T, view_point_obj_diff.T).T
        obj_pcd = env.affordance_pcd[i].reshape(200, 3).cpu().numpy()
        directions = obj_pcd - view_point_obj
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        # Perform ray-tracing to determine visible points on the object
        locations, index_ray, index_tri = env.aff_mesh[i].ray.intersects_location(ray_origins=view_point_obj,
                                                                                  ray_directions=directions,
                                                                                  multiple_hits=False)
        if locations.shape != (200, 3):
            expanded_locations = np.zeros((200, 3))
            expanded_locations[:, :] = locations[0, :]
            expanded_locations[:locations.shape[0], :] = locations
            locations = expanded_locations
        visible_points_obj[i, :] = locations
        visible_points_w[i, :] = np.matmul(obj_mat_single, locations.T).T + obj_pose_reset[i, :3]

        obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200, 3), axis=0)

        top_grasp = cfg['environment']['top']
        # Get the x_dir of the grasping frame
        if top_grasp:
            hand_dir_x_w = np.zeros((1, 3))
            hand_dir_x_w[0, 2] = 1
        else:
            hand_dir_x_w = heading_dir_start_w - obj_aff_center_in_w
            hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)

        # Get wrist position and orientation
        pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w

        rot_mats, projection_lengths = sample_rot_mats(hand_dir_x_w, sample_num, visible_points_w[i])

        # Check if IK is feasible
        feasible_ik_flag = np.zeros((sample_num), dtype='bool')
        ik_results = np.zeros((sample_num, 6), dtype='float32')
        for j in range(sample_num):
            rot_mat = rot_mats[j]
            wrist_in_world = rot_mat
            hand_center_in_world = np.matmul(wrist_in_world, hand_center.T).T

            pos_in_ur5 = np.zeros((3, 1))
            pos_in_ur5[0, 0] = pos[0, 0] - 0. + hand_center_in_world[0, 0]
            pos_in_ur5[1, 0] = pos[0, 1] - 0. + hand_center_in_world[0, 1]
            pos_in_ur5[2, 0] = pos[0, 2] - 0.771 + hand_center_in_world[0, 2]
            pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)
            
            wrist_mat_in_ur5 = np.matmul(ur5_to_world.T, wrist_in_world)

            gd = np.eye(4)
            gd[:3, :3] = wrist_mat_in_ur5
            gd[0, 3] = pos_in_ur5_new[0, 0]
            gd[1, 3] = pos_in_ur5_new[1, 0]
            gd[2, 3] = pos_in_ur5_new[2, 0]

            ik_result = ik.findClosestIK(gd, theta0)
            if ik_result is None or np.isnan(ik_result).any():
                feasible_ik_flag[j] = False
                continue
            else:
                ik_results[j, :] = ik_result
                feasible_ik_flag[j] = True

        # Select the best IK solution
        feasible_indices = np.where(feasible_ik_flag)[0]
        scores = np.ones(sample_num, dtype='float32')
        scores = scores * 10000.
        if min(projection_lengths) < 0.18:
            for j in feasible_indices:
                if projection_lengths[j] < 0.18:
                    score1 = projection_lengths[j] * cfg['environment']['length_score_coeff']
                    score2 = abs(ik_results[j, 4] - 1.57) * cfg['environment']['angle_score_coeff']
                    score3 = (abs(ik_results[j, 4]) - 3.2) * cfg['environment']['angle_score_coeff'] * 0.5
                    scores[j] = score1 + score2 + score3
                else:
                    scores[j] = 10000.
            best_index = np.argmin(scores)
            qpos_reset_r[i, :6] = ik_results[best_index]
        else:
            best_index = np.argmin(projection_lengths)
            qpos_reset_r[i, :6] = ik_results[best_index]
        
    # Check self-collision
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
        if len(false_indices) > 0:
            chosen_index = np.random.choice(false_indices)
            qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
            obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
        else:
            qpos_reset_r[true_idx, :6] = [angle + np.pi / 2 - 0.3, -1.57, 1.57, 0., 1.57, -1.57]
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
    aff_vec_new, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
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
        aff_vec = aff_vec_new.astype('float32')

        action_r, student_mlp_obs = dagger.act(obs_r, student_driven_ratio, aff_vec, aff_vec_dim)

        # If in evaluation mode and grasp phase is completed, enter lift phase
        if is_evaluation and step >= n_steps_r:
            # In lift phase, use fixed arm pose
            action_r[:, :6] = theta0
            if step == n_steps_r:
                # Enable root guidance at the start of lifting
                env.switch_root_guidance(True)
                print("Starting lift phase...", file=sys.stdout)

        reward_r, _, dones = env.step(action_r.astype('float32'), np.zeros_like(action_r).astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        obs_new_r = obs_new_r[:].astype('float32')
        aff_vec_new, _ = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))

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
        table_reward_r = -np.sum(np.log(50 * np.clip(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87], a_min=0.002, a_max=0.02)) * finger_weights, axis=1)
        arm_height_reward_r = -np.sum(np.log(50 * np.clip(obs_new_r[:, -ob_dim_r+89:-ob_dim_r+93], a_min=0.002, a_max=0.02)), axis=1)
        
        one_check = global_state[:, 124:128]
        arm_collision_reward_r = np.sum(one_check, axis=1)

        for i in range(num_envs):
            rewards_r[i]['affordance_reward'] = affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff']
            rewards_r[i]['table_reward'] = table_reward_r[i] * cfg['environment']['reward']['table_reward']['coeff']
            rewards_r[i]['arm_height_reward'] = arm_height_reward_r[i] * cfg['environment']['reward']['arm_height_reward']['coeff']
            rewards_r[i]['arm_collision_reward'] = arm_collision_reward_r[i] * cfg['environment']['reward']['arm_collision_reward']['coeff']
            
            rewards_r[i]['reward_sum'] = (
                        rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] +
                        rewards_r[i]['table_reward'] + rewards_r[i]['arm_height_reward'] + rewards_r[i]['arm_collision_reward'])
            reward_r[i] = rewards_r[i]['reward_sum']
        reward_r.clip(min=reward_clip)

        for i in range(len(rewards_r_sum)):
            for k in rewards_r_sum[i].keys():
                rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]

        # Only collect training data in non-evaluation mode
        if not is_evaluation:
            obs_r_student = np.concatenate([obs_r[:, :tobeEncode_dim*t_steps], student_mlp_obs], axis=1)
            dagger.step(total_obs=obs_r_student, rews=reward_r, dones=dones, value_obs=obs_r[:, -ob_dim_r:])

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
    value_obs = obs_r[:, -ob_dim_r:]

    # Only update policy in non-evaluation mode
    if not is_evaluation:
        prop_mse_loss, action_mse_loss = dagger.update(value_obs)

    actor_student_r.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    end = time.time()

    ave_reward = {}
    for k in rewards_r_sum[0].keys():
        ave_reward[k] = 0
    for k in rewards_r_sum[0].keys():
        for i in range(len(rewards_r_sum)):
            ave_reward[k] = ave_reward[k] + rewards_r_sum[i][k]
        # When calculating average rewards, only consider grasp phase steps
        ave_reward[k] = ave_reward[k] / (len(rewards_r_sum) * n_steps_r)
    ave_reward['recon_loss'] = prop_mse_loss
    ave_reward['action_loss'] = action_mse_loss

    # Only log training rewards in non-evaluation mode
    if args.log_name is not None and not is_evaluation:
        wandb.log(ave_reward, step=update)

    if args.log_name is None:
        print(ave_reward, file=sys.stdout)

    print('----------------------------------------------------', file=sys.stdout)
    print('{:>6}th iteration'.format(update), file=sys.stdout)
    print('{:<40} {:>6}'.format("average reward: ", '{:0.10f}'.format(ave_reward['reward_sum'])), file=sys.stdout)
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)), file=sys.stdout)
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_r / (end - start))), file=sys.stdout)
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps_r / (end - start)
                                                                * cfg['environment']['control_dt'])), file=sys.stdout)
    print('{:<40} {:>6}'.format("prop mse loss: ", '{:0.10f}'.format(prop_mse_loss)), file=sys.stdout)
    print('{:<40} {:>6}'.format("action mse loss: ", '{:0.10f}'.format(action_mse_loss)), file=sys.stdout)
    print('----------------------------------------------------\n', file=sys.stdout)