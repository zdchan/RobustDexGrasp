#!/usr/bin/python
import os
import sys
import time
import random
import argparse
import numpy as np
from random import choice

# Torch related imports
import torch
import torch.nn as nn

# RaiSim related imports
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_student as hand
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
from raisimGymTorch.env.bin.allegro_student import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats
from raisimGymTorch.helper import rotations
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5
import raisimGymTorch.algo.ppo_dagger_recon.module as ppo_module

# Other imports
from scipy.spatial.transform import Rotation as R

# Enable line buffering for real-time output printing
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

# ===== Configuration Parameters =====
exp_name = "student"
weight_path_student = 'student_ckpt/full_5500_r.pt'

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_path_student)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-ln', '--log_name', type=str, default='single_obj')

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

# Load configuration
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

# Set random seed
if args.seed != 1:
    cfg['seed'] = args.seed

# Environment settings
cfg['environment']['visualize'] = False

# ===== Object Loading Setup =====
# Set dataset for quantitative evaluation
# cat_name = 'new_training_set'
cat_name = 'shapenet-30obj'


# Whether should load stable states
if cat_name == 'shapenet-30obj':
    stable = True
else:
    stable = False

# Number of repetitions per object
repeat_per_obj = 1

# Set loading set
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path, file=sys.stdout)

# Get all items in the directory
items = os.listdir(directory_path)

# Filter out only the folders
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

# Build object lists
obj_list = []
obj_path_list = []
obj_ori_list = folder_names

# Create final object list based on repetition count
num_envs = len(obj_ori_list) * repeat_per_obj
for i in range(repeat_per_obj):
    for item in obj_ori_list:
        obj_list.append(item)
        
# ===== Network and Environment Setup =====
activations = nn.LeakyReLU
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs, file=sys.stdout)

# Domain randomization settings
if not cfg['environment']['randomization_eval']:
    print("no randomization", file=sys.stdout)
    cfg['environment']['hardware']['randomize_friction'] = "0.8"
    cfg['environment']['hardware']['randomize_gains_hand_p'] = 0.
    cfg['environment']['hardware']['randomize_gains_hand_d'] = 0.
    cfg['environment']['hardware']['randomize_gains_arm_p'] = 0.
    cfg['environment']['hardware']['randomize_gains_arm_d'] = 0.
    cfg['environment']['hardware']['randomize_gc_hand'] = 0.
    cfg['environment']['hardware']['randomize_gc_arm'] = 0.
    cfg['environment']['hardware']['randomize_frame_position'] = 0.
    cfg['environment']['hardware']['randomize_frame_orientation'] = 0.
else:
    print("randomization", file=sys.stdout)

# Create environment
env = VecEnv(obj_list, hand.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished", file=sys.stdout)

# Load multi-articulated objects
for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)

# ===== Model Dimension Setup =====
ob_dim_r = 153
act_dim = 22
print('ob dim', ob_dim_r, file=sys.stdout)
print('act dim', act_dim, file=sys.stdout)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim = 26
aff_vec_dim = 51
total_obs_dim = tobeEncode_dim * t_steps + ob_dim_r

# Training parameters
grasp_steps = cfg['environment']['grasp_steps'] + 30
lift_steps = 100
n_steps_r = grasp_steps + lift_steps
total_steps_r = n_steps_r * env.num_envs

# ===== Build Neural Network Models =====
# Student actor network
actor_student_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

# Property latent encoder
prop_latent_encoder = ppo_module.LSTM_StateHistoryEncoder(tobeEncode_dim, prop_latent_dim, t_steps, device)

# Critic network
critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

# ===== Load Model Weights =====
test_dir = True
saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[], test_dir=test_dir)

print(f"load weight from {saver.data_dir.split('eval')[0] + weight_path_student}", file=sys.stdout)

checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device('cpu'))
actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

# ===== Load Object Data =====
# Load lowest point data
lowest_points = np.zeros((num_envs, 1), dtype='float32')
if stable:
    stable_states = np.zeros((num_envs, 7), dtype='float32')    
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())
    if stable:
        stable_state_path = home_path + f"/rsc/{cat_name}/{obj_list[i]}/{obj_list[i]}.npy"
        stable_states[i] = np.load(stable_state_path)[-1, :7]
        
# ===== Evaluation =====
success_rate = 0.0

# Create dictionary to track success/failure statistics for each object
object_failure_stats = {}
for obj_name in obj_list:
    if obj_name not in object_failure_stats:
        object_failure_stats[obj_name] = {"failures": 0, "attempts": 0}

# Run multiple evaluations
for update in range(5):
    start = time.time()

    # Initialize reset states
    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)

    # Set initial finger positions
    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']

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

    # Initial joint angles and weights
    theta0 = [0.0, -1.57, 1.57, 0., 1.57, -1.57]
    joint_weights = [1, 1, 1, 1, 1, 1]

    # Initialize UR5 inverse kinematics
    ik = InverseKinematicsUR5()
    ik.setJointWeights(joint_weights)
    ik.setJointLimits(-3.14, 3.14)

    # Visible points setup
    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    # Number of IK samples to try
    sample_num = cfg['environment']['sample_num']

    # Calculate initial positions and poses for each environment
    for i in range(num_envs):
        # Randomly sample object position
        while True:
            angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
            distance = np.random.uniform(0.45, 0.75)
            sample_x = distance * np.cos(angle)
            sample_y = distance * np.sin(angle)
            if sample_x < 0.25 and sample_x > -0.25:
                break

        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y
        
        if stable:
            obj_pose_reset[i, 2:7] = stable_states[i, 2:7]
            obj_pose_reset[i, 2] += 0.005
            quats = stable_states[i, 3:7]
        else:
            obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
            obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

            axis_angles = np.zeros((1, 3))
            axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
            quats = rotations.axisangle2quat(axis_angles)
            obj_pose_reset[i, 3:7] = quats

        # Get partial point cloud (not relevant for hardware deployment)
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
        # Get x-direction of the grasping frame
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

    # Check for self-collision
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
    
    # Handle self-collision cases
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
            qpos_reset_r[true_idx, :6] = [angle+np.pi/2-0.3, -1.57, 1.57, 0., 1.57, -1.57]
            obj_pose_reset[true_idx, 0] = 0.1
            obj_pose_reset[true_idx, 1] = -0.5

    # Reset environment state
    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    # Get observations
    obs_new_r, dis_info = env.observe_vision_new()
    aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
    env.update_target(target_center)

    final_actions = np.zeros((num_envs, act_dim), dtype='float32')

    # Apply bias cases
    biased = cfg['environment']['biased']
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')

    # Run grasp and lift steps
    for step in range(n_steps_r):
        frame_start = time.time()

        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')

        # Get encoded observations
        encode_obs = torch.from_numpy(obs_r[:, :tobeEncode_dim * t_steps]).to(device)

        # Get student latent representation and build MLP observation
        student_latent = prop_latent_encoder(encode_obs)
        student_mlp_obs = torch.cat((torch.from_numpy(obs_r[:, -ob_dim_r:-ob_dim_r + tobeEncode_dim]),
                                     student_latent.cpu(),
                                     torch.from_numpy(obs_r[:, -ob_dim_r + tobeEncode_dim + prop_latent_dim:-aff_vec_dim]),
                                     torch.from_numpy(aff_vec)), dim=1).to(device)

        # Get action
        action_r = actor_student_r.architecture.architecture(student_mlp_obs.to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)

        # Execute grasp or lift action
        if step < grasp_steps:
            final_actions = action_r
        else:
            action_r = final_actions
            action_r[:, :6] = theta0
            if step == grasp_steps:
                env.switch_root_guidance(True)

        # Execute environment step
        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        # Get new observations
        obs_new_r, dis_info = env.observe_vision_new()
        aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))

        # Handle bias cases
        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)

    # Evaluate grasp success rate
    global_state = env.get_global_state()
    lifted = global_state[:, 107] - obj_pose_reset[:, 2] > 0.1
    print("current success rate", np.sum(lifted) / num_envs, file=sys.stdout)

    success_rate = (update * success_rate + np.sum(lifted) / num_envs) / (update + 1)
    print("average success rate", success_rate, file=sys.stdout)

    # Update failure statistics for each object
    for i in range(num_envs):
        obj_name = obj_list[i]
        object_failure_stats[obj_name]["attempts"] += 1
        if not lifted[i]:
            object_failure_stats[obj_name]["failures"] += 1

    # Print names of failed objects
    failed_indices = np.where(lifted == 0)[0]
    if len(failed_indices) > 0:
        print("Failed objects:", file=sys.stdout)
        for idx in failed_indices:
            print(f"  - {obj_list[idx]}", file=sys.stdout)
    else:
        print("All objects were successfully grasped!", file=sys.stdout)


# ===== Output Statistics =====
print("\n===== Object Failure Statistics =====", file=sys.stdout)
print(f"{'Object Name':<30} {'Failures':<10} {'Attempts':<10} {'Failure Rate (%)':<20}", file=sys.stdout)
print("-" * 70, file=sys.stdout)

# Sort by failure rate from high to low
sorted_stats = sorted(object_failure_stats.items(), 
                     key=lambda x: x[1]["failures"] / x[1]["attempts"] if x[1]["attempts"] > 0 else 0, 
                     reverse=True)

for obj_name, stats in sorted_stats:
    failure_rate = (stats["failures"] / stats["attempts"] * 100) if stats["attempts"] > 0 else 0
    print(f"{obj_name:<30} {stats['failures']:<10} {stats['attempts']:<10} {failure_rate:.2f}%", file=sys.stdout)

# Calculate total attempts and total failures
total_attempts = sum(stats["attempts"] for stats in object_failure_stats.values())
total_failures = sum(stats["failures"] for stats in object_failure_stats.values())
total_success_rate = ((total_attempts - total_failures) / total_attempts * 100) if total_attempts > 0 else 0

print("\nTotal success rate: {:.2f}%".format(total_success_rate), file=sys.stdout)





