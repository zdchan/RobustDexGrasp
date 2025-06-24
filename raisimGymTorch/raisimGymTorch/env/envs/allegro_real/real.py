#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_real as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
from raisimGymTorch.env.bin.allegro_real import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats
from random import choice
import os
import math
import time
import raisimGymTorch.algo.ppo_dagger_recon.module as ppo_module
import torch.nn as nn
import numpy as np
import torch
import argparse
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5
import torch
from raisimGymTorch.env.hardware.realsense.PointCloud import Realsense

weight_path_student = 'student_ckpt/full_5500_r.pt'
# task specification
task_name = "student"
# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + '/cfgs/cfg_reg.yaml', 'r'))

num_envs = 1
cfg['environment']['visualize'] = True
cfg['environment']['num_envs'] = num_envs
cfg['environment']['num_threads'] = 1

# Environment definition
env = VecEnv(['dummy'], mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], cat_name=cfg['environment']['load_set'])

ob_dim_r = 153
# act_dim = env.num_acts
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim=26
aff_vec_dim = 51
total_obs_dim = tobeEncode_dim*t_steps + ob_dim_r

# Training
reward_clip = -2.0
grasp_steps = cfg['environment']['grasp_steps']
lift_steps = 1
n_steps_r = grasp_steps + lift_steps
total_steps_r = n_steps_r * env.num_envs

# RL network

actor_student_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
prop_latent_encoder = ppo_module.LSTM_StateHistoryEncoder(tobeEncode_dim, prop_latent_dim, t_steps, device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim_r, 1), device)


test_dir = True

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data_all/" + task_name,
                           save_items=[], test_dir=test_dir)

checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device('cpu'))
actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

rs = Realsense(cfg['environment']['hardware']['pointcloud_real']['camera_K_path'], 200)

while True:
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')
    target_center = np.zeros_like(env.affordance_center)
    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']

    hand_center_sample_w = np.zeros((1, 3))
    hand_center_sample_w[0, 0] = cfg['environment']['camera_position'][0]
    hand_center_sample_w[0, 1] = cfg['environment']['camera_position'][1]
    hand_center_sample_w[0, 2] = cfg['environment']['camera_position'][2]

    wrist_bias = np.zeros((1, 3))
    wrist_bias[0, 0] = -0.0091
    wrist_bias[0, 2] = -0.095

    ur5_to_world = np.eye(3)
    ur5_to_world[0, 0] = 0
    ur5_to_world[0, 1] = -1
    ur5_to_world[1, 0] = 1
    ur5_to_world[1, 1] = 0

    theta0 = [0.0, -1.57, 1.57, 0., 1.57, -1.57]
    joint_weights = [1, 1, 1, 1, 1, 1]

    ik = InverseKinematicsUR5()
    ik.setJointWeights(joint_weights)
    ik.setJointLimits(-3.14, 3.14)

    sample_num = cfg['environment']['sample_num']

    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    obj_pos_mean, obj_pointcloud = rs.GetPointCloud()
    obj_init_xyz_qwxyz = np.array([obj_pos_mean[0][0], obj_pos_mean[0][1], obj_pos_mean[0][1], 0.707, 0, 0.707, 0])

    obj_pose_reset[0, :7] = obj_init_xyz_qwxyz # mean of pointcloud
    visible_points_w[0, :] = obj_pointcloud # sample randomly from RGBD in mask
    angle = math.atan2(obj_init_xyz_qwxyz[1], obj_init_xyz_qwxyz[0])

    # get the x_dir of the grasping frame
    obj_aff_center_in_w = np.mean(visible_points_w[0].reshape(200,3), axis=0)
    
    top_grasp = cfg['environment']['top']
    # get the x_dir of the grasping frame
    if top_grasp:
        hand_dir_x_w = np.zeros((1, 3))
        hand_dir_x_w[0, 2] = 1
    else:
        hand_dir_x_w = hand_center_sample_w - obj_aff_center_in_w
        hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)
    pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w

    rot_mats, projection_lengths = sample_rot_mats(hand_dir_x_w, sample_num, visible_points_w[0])

    feasible_ik_flag = np.zeros((sample_num), dtype='bool')
    ik_results = np.zeros((sample_num, 6), dtype='float32')
    for j in range(sample_num):
        rot_mat = rot_mats[j]
        wrist_in_world = rot_mat
        wrist_bias_in_world = np.matmul(wrist_in_world, wrist_bias.T).T
        pos_in_ur5 = np.zeros((3, 1))
        pos_in_ur5[0, 0] = pos[0, 0] - 0. + wrist_bias_in_world[0, 0]
        pos_in_ur5[1, 0] = pos[0, 1] - 0. + wrist_bias_in_world[0, 1]
        pos_in_ur5[2, 0] = pos[0, 2] - 0.771 + wrist_bias_in_world[0, 2]
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
            qpos_reset_r[0, :6] = ik_result
            collision_check = env.check_collision(qpos_reset_r)
            if not collision_check:
                feasible_ik_flag[j] = False
                continue
            else:
                #print(f"feasible ik in {j}")
                feasible_ik_flag[j] = True
                ik_results[j] = ik_result
    feasible_indices = np.where(feasible_ik_flag)[0]
    scores = np.ones(sample_num, dtype='float32')
    scores = scores * 10000.
    if min(projection_lengths) < 0.15:
        for j in feasible_indices:
            if projection_lengths[j] < 0.15:
                score1 = projection_lengths[j] * cfg['environment']['length_score_coeff']
                score2 = abs(ik_results[j, 4] - 1.57) * cfg['environment']['angle_score_coeff']
                score3 = (abs(ik_results[j, 4]) - 3.2) * cfg['environment']['angle_score_coeff'] * 0.5
                scores[j] = score1 + score2 + score3
            else:
                scores[j] = 10000.
        best_index = np.argmin(scores)
        qpos_reset_r[0, :6] = ik_results[best_index]
    else:
        best_index = np.argmin(projection_lengths)
        qpos_reset_r[0, :6] = ik_results[best_index]

    vis_point = visible_points_w.reshape(200*3, -1).astype('float32')
    env.set_sample_point_visual(vis_point)

    grasp_steps = cfg['environment']['grasp_steps']
    n_steps_r = grasp_steps + lift_steps

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset
                    )

    obs_new_r, aff_vec = env.observe_student_deploy(torch.from_numpy(visible_points_w).to(device))
    aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
    env.set_joint_sensor_visual(show_point)

    final_actions = np.zeros((num_envs, act_dim), dtype='float32')

    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')
        encode_obs = torch.from_numpy(obs_r[:, :tobeEncode_dim * t_steps]).to(device)
        student_latent = prop_latent_encoder(encode_obs)
        student_mlp_obs = torch.cat((torch.from_numpy(obs_r[:, -ob_dim_r:-ob_dim_r + tobeEncode_dim]),
                                    student_latent.cpu(), 
                                    torch.from_numpy(obs_r[:, -ob_dim_r + tobeEncode_dim + prop_latent_dim:-aff_vec_dim]),
                                    torch.from_numpy(aff_vec)), dim=1).to(device)

        action_r = actor_student_r.architecture.architecture(student_mlp_obs.to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)
        if step < grasp_steps:
            final_actions = action_r
        else:
            action_r = final_actions
            action_r[:, :6] = theta0
            if step == grasp_steps:
                print("lift")
                env.switch_root_guidance(True)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, aff_vec = env.observe_student_deploy(torch.from_numpy(visible_points_w).to(device))
        step = step + 1