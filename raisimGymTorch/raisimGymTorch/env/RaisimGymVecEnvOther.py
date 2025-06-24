# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import faulthandler;

# from fontTools.merge.util import current_time

faulthandler.enable()

import numpy as np
import platform
import os
from scipy.spatial.transform import Rotation as R
import torch
import trimesh

from raisimGymTorch.helper import rotations
class RaisimGymVecEnvTest:

    def __init__(self, obj_list, impl, cfg, normalize_ob=False, seed=0, normalize_rew=True, clip_obs=10., cat_name=None, cent_training=False, two_hand=False):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'

        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs_r = self.wrapper.getRightObDim()
        self.num_obs_l = self.wrapper.getLeftObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.num_gs = self.wrapper.getGSDim()
        self._observation_r = np.zeros([self.num_envs, self.num_obs_r], dtype=np.float32)
        self._observation_l = np.zeros([self.num_envs, self.num_obs_l], dtype=np.float32)
        self._global_state = np.zeros([self.num_envs, self.num_gs], dtype=np.float32)
        self._global_state_l = np.zeros([self.num_envs, self.num_gs], dtype=np.float32)
        self.obs_rms_r = RunningMeanStd(shape=[self.num_envs, self.num_obs_r])
        self.obs_rms_l = RunningMeanStd(shape=[self.num_envs, self.num_obs_l])
        self.gs_rms = RunningMeanStd(shape=[self.num_envs, self.num_gs])
        self._reward_r = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_l = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]

        self.affordance_pcd = np.zeros([len(obj_list),200,3])
        self.affordance_normals = np.zeros([len(obj_list),200,3])
        self.aff_mesh = [None] * len(obj_list)
        self.non_affordance_pcd = np.zeros([len(obj_list),200,3])
        self.non_affordance_normals = np.zeros([len(obj_list),200,3])
        self.non_aff_mesh = [None] * len(obj_list)
        self.affordance_center = np.zeros((len(obj_list),3), 'float32')
        self.non_affordance_center = np.zeros((len(obj_list),3), 'float32')

        for obj_name in np.unique(obj_list):
            if two_hand == False:
                aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/top_watertight_tiny.obj'
                non_aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/bottom_watertight_tiny.obj'
            else:
                aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/bottom_watertight_tiny.obj'
                non_aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/top_watertight_tiny.obj'
            aff_mesh = trimesh.load_mesh(aff_mesh_name)
            aff_points, aff_face_id = trimesh.sample.sample_surface(aff_mesh, 200)
            aff_normals = aff_mesh.face_normals[aff_face_id]

            aff_center = aff_mesh.centroid

            non_aff_mesh = trimesh.load_mesh(non_aff_mesh_name)
            non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
            non_aff_normals = non_aff_mesh.face_normals[non_aff_face_id]

            non_aff_center = non_aff_mesh.centroid
            if non_aff_mesh.vertices.shape[0] < 25:
                not_two_parts = True
            else:
                not_two_parts = False

            for i in range(len(obj_list)):
                if obj_list[i] == obj_name:
                    self.affordance_pcd[i] = aff_points
                    self.affordance_normals[i] = aff_normals
                    self.aff_mesh[i] = aff_mesh
                    self.non_affordance_pcd[i] = non_aff_points
                    self.non_affordance_normals[i] = non_aff_normals
                    self.non_aff_mesh[i] = non_aff_mesh
                    self.affordance_center[i] = aff_center
                    if not not_two_parts:
                        self.non_affordance_center[i] = non_aff_center
                    else:
                        self.non_affordance_center[i, :] = 100

        self.affordance_pcd = torch.tensor(self.affordance_pcd).float().to('cuda')
        self.affordance_normals = torch.tensor(self.affordance_normals).float().to('cuda')
        self.non_affordance_pcd = torch.tensor(self.non_affordance_pcd).float().to('cuda')
        self.non_affordance_normals = torch.tensor(self.non_affordance_normals).float().to('cuda')


    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_pd_wrist(self):
        self.wrapper.set_pd_wrist()

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action_r, action_l):
        self.wrapper.step(action_r, action_l, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()

    def step2(self, action_r, action_l):
        self.wrapper.step2(action_r, action_l, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()

    def reset_right_hand(self, obj_pose_step_r, hand_ee_step_r, hand_pose_step_r):
        self.wrapper.reset_right_hand(obj_pose_step_r, hand_ee_step_r, hand_pose_step_r)

    def step_imitate(self, action_r, action_l, obj_pose_r, hand_ee_r, hand_pose_r, obj_pose_l, hand_ee_l, hand_pose_l, imitate_right, imitate_left):
        self.wrapper.step_imitate(action_r, action_l, obj_pose_r, hand_ee_r, hand_pose_r, obj_pose_l, hand_ee_l, hand_pose_l, imitate_right, imitate_left, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5, cent_training=False):
        mean_file_name_r = dir_name + "/mean_r" + str(iteration) + ".csv"
        var_file_name_r = dir_name + "/var_r" + str(iteration) + ".csv"
        mean_file_name_l = dir_name + "/mean_l" + str(iteration) + ".csv"
        var_file_name_l = dir_name + "/var_l" + str(iteration) + ".csv"
        if cent_training:
            mean_file_name_g = dir_name + "/mean_g" + str(iteration) + ".csv"
            var_file_name_g = dir_name + "/var_g" + str(iteration) + ".csv"
        self.obs_rms_r.count = count
        self.obs_rms_r.mean = np.loadtxt(mean_file_name_r, dtype=np.float32)
        self.obs_rms_r.var = np.loadtxt(var_file_name_r, dtype=np.float32)
        if os.path.exists(mean_file_name_l) and os.path.exists(var_file_name_l):
            self.obs_rms_l.count = count
            self.obs_rms_l.mean = np.loadtxt(mean_file_name_l, dtype=np.float32)
            self.obs_rms_l.var = np.loadtxt(var_file_name_l, dtype=np.float32)
        if cent_training:
            self.gs_rms.count = count
            self.gs_rms.mean = np.loadtxt(mean_file_name_g, dtype=np.float32)
            self.gs_rms.var = np.loadtxt(var_file_name_g, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name_r = dir_name + "/mean_r" + iteration + ".csv"
        var_file_name_r = dir_name + "/var_r" + iteration + ".csv"
        mean_file_name_l = dir_name + "/mean_l" + iteration + ".csv"
        var_file_name_l = dir_name + "/var_l" + iteration + ".csv"
        np.savetxt(mean_file_name_r, self.obs_rms_r.mean)
        np.savetxt(var_file_name_r, self.obs_rms_r.var)
        np.savetxt(mean_file_name_l, self.obs_rms_l.mean)
        np.savetxt(var_file_name_l, self.obs_rms_l.var)   

    def euler_to_rotation_matrix(self, euler_angles):
        """Convert Euler angles to rotation matrices."""
        batch_size = euler_angles.shape[0]
        c1 = torch.cos(euler_angles[:, 0])
        s1 = torch.sin(euler_angles[:, 0])
        c2 = torch.cos(euler_angles[:, 1])
        s2 = torch.sin(euler_angles[:, 1])
        c3 = torch.cos(euler_angles[:, 2])
        s3 = torch.sin(euler_angles[:, 2])

        rotation_matrices = torch.zeros((batch_size, 3, 3), device=euler_angles.device)
        rotation_matrices[:, 0, 0] = c2 * c3
        rotation_matrices[:, 0, 1] = -c2 * s3
        rotation_matrices[:, 0, 2] = s2
        rotation_matrices[:, 1, 0] = c1 * s3 + c3 * s1 * s2
        rotation_matrices[:, 1, 1] = c1 * c3 - s1 * s2 * s3
        rotation_matrices[:, 1, 2] = -c2 * s1
        rotation_matrices[:, 2, 0] = s1 * s3 - c1 * c3 * s2
        rotation_matrices[:, 2, 1] = c3 * s1 + c1 * s2 * s3
        rotation_matrices[:, 2, 2] = c1 * c2

        return rotation_matrices    

    def observe_vision_new(self):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 54:105].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, self.affordance_pcd)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(self.affordance_pcd, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        obj_euler_wrist = torch.from_numpy(global_state[:, :3]).to('cuda')
        obj_euler_world = torch.from_numpy(global_state[:, 118:121]).to('cuda')

        r_obj = self.euler_to_rotation_matrix(obj_euler_world).unsqueeze(1).repeat(1, joints.shape[1], 1, 1).to('cuda')
        af_vec_rotated = torch.matmul(r_obj, af_vec.reshape(num_envs, -1, 3).to('cuda').unsqueeze(-1)).squeeze(-1)
        af_vec = af_vec_rotated.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        obs_r = np.concatenate([obs_r, af_vec], axis=-1)

        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')
        dis_info = np.concatenate([min_dis_af.cpu().numpy(), show_af_point], axis=-1)

        return obs_r, dis_info


    def observe_student_aff(self, visible_points):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 128:179].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, visible_points)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(visible_points, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        af_vec = af_vec.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')

        return af_vec, show_af_point


    def observe_student_deploy(self, visible_points):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 128:179].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, visible_points)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(visible_points, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        af_vec = af_vec.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        obs_r = np.concatenate([obs_r, af_vec], axis=-1)

        return obs_r, af_vec


    def get_global_state(self, update_mean=True):
        self.wrapper.get_global_state(self._global_state)

        if self.normalize_ob:
            if update_mean:
                self.gs_rms.update(self._global_state)

            return self._normalize_global_state(self._global_state)
        else:
            return self._global_state.copy()

    def get_global_state_l(self, update_mean=True):
        self.wrapper.get_global_state_l(self._global_state_l)

        if self.normalize_ob:
            if update_mean:
                self.gs_rms.update(self._global_state_l)

            return self._normalize_global_state(self._global_state_l)
        else:
            return self._global_state_l.copy()

    def set_rootguidance(self):
        self.wrapper.set_rootguidance()

    def switch_root_guidance(self, obj_pos_bias):
        self.wrapper.switch_root_guidance(obj_pos_bias)

    def switch_obj_pos(self, is_on):
        self.wrapper.switch_obj_pos(is_on)

    def control_switch(self, left, right):
        self.wrapper.control_switch(left, right)

    def control_switch_all(self, left, right):
        self.wrapper.control_switch_all(left, right)

    def reset(self):
        self._reward_r = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_l = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def add_stage(self, stage_dim, stage_pos):
        self.wrapper.add_stage(stage_dim, stage_pos)

    def switch_arctic(self, idx):
        self.wrapper.switch_arctic(idx)

    def load_object(self, obj_idx, obj_weight, obj_dim, obj_type):
        self.wrapper.load_object(obj_idx, obj_weight, obj_dim, obj_type)

    def load_articulated(self, obj_model):
        self.wrapper.load_articulated(obj_model)

    def load_multi_articulated(self, obj_models):
        self.wrapper.load_multi_articulated(obj_models)

    def reset_state(self, init_state_r, init_state_l, init_vel_r, init_vel_l, obj_pose):
        self.wrapper.reset_state(init_state_r, init_state_l, init_vel_r, init_vel_l, obj_pose)

    def set_goals_r(self, obj_pos_r, ee_pos_r, pose_r, qpos_r):
        self.wrapper.set_goals_r(obj_pos_r, ee_pos_r, pose_r, qpos_r)

    def set_imitation_goals(self, pose_l, pose_r, obj_pose):
        self.wrapper.set_imitation_goals(pose_l, pose_r, obj_pose)

    def set_goals_r2(self, obj_pos_r, ee_pos_r, pose_r, qpos_r, contact_r):
        self.wrapper.set_goals_r2(obj_pos_r, ee_pos_r, pose_r, qpos_r, contact_r)

    def set_ext(self, ext_force, ext_torque):
        self.wrapper.set_ext(ext_force, ext_torque)

    def set_pregrasp(self, obj_pos, ee_pos, pose):
        self.wrapper.set_pregrasp(obj_pos, ee_pos, pose)

    def set_goals(self, obj_angle, obj_pos, ee_pos_r, ee_pos_l, pose_r, pose_l, qpos_r, qpos_l, contact_r, contact_l):
        self.wrapper.set_goals(obj_angle, obj_pos, ee_pos_r, ee_pos_l, pose_r, pose_l, qpos_r, qpos_l, contact_r, contact_l)

    def update_target(self, target_center):
        self.wrapper.update_target(target_center)

    def set_joint_sensor_visual(self, joint_sensor_visual):
        self.wrapper.set_joint_sensor_visual(joint_sensor_visual)

    def set_sample_point_visual(self, joint_sensor_visual):
        self.wrapper.set_sample_point_visual(joint_sensor_visual)

    def check_collision(self, joint_state):
        return self.wrapper.check_collision(joint_state)

    def set_joint_sensor_visual_l(self, joint_sensor_visual):
        self.wrapper.set_joint_sensor_visual_l(joint_sensor_visual)

    def set_obj_goal(self, obj_angle, obj_pos):
        self.wrapper.set_obj_goal(obj_angle, obj_pos)

    def _normalize_observation(self, obs, is_rhand):
        if self.normalize_ob:
            if is_rhand:
                return np.clip((obs - self.obs_rms_r.mean) / np.sqrt(self.obs_rms_r.var + 1e-8), -self.clip_obs,
                               self.clip_obs)
            else:
                return np.clip((obs - self.obs_rms_l.mean) / np.sqrt(self.obs_rms_l.var + 1e-8), -self.clip_obs,
                               self.clip_obs)
        else:
            return obs

    def _normalize_global_state(self, gs):
        if self.normalize_ob:
            return np.clip((gs - self.gs_rms.mean) / np.sqrt(self.gs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return gs

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def get_reward_info_l(self):
        reward_info = self.wrapper.rewardInfoLeft()
        return reward_info

    def get_reward_info_r(self):
        reward_info = self.wrapper.rewardInfoRight()
        return reward_info

    def get_pca_rewards(self, obs, is_right):
        num_envs = obs.shape[0]
        eulers = obs.reshape(num_envs,-1, 3).copy()

        eulers = eulers.reshape(-1,3)
        rotvec = R.from_euler('XYZ', eulers, degrees=False)
        rotvec = rotvec.as_rotvec().reshape(num_envs,-1)

        if is_right:
            joint_pca = np.matmul(rotvec, np.linalg.inv(self.th_comps_r))
            pca_target = self.mean_pca_r.repeat(num_envs, 0)

        else:
            joint_pca = np.matmul(rotvec, np.linalg.inv(self.th_comps_l))
            pca_target = self.mean_pca_l.repeat(num_envs, 0)

        joint_pca_norm = joint_pca / (np.linalg.norm(joint_pca, axis=-1, keepdims=True)+1e-5)
        pca_target_norm = pca_target / (np.linalg.norm(pca_target, axis=-1, keepdims=True)+1e-5)
        pca_dist = joint_pca_norm * pca_target_norm
        cos_sim = 1 - pca_dist.sum(-1)
        cos_sim[cos_sim<0.4] *= 0.1
        return cos_sim

    def debugShowObs(self, obs):
        self.wrapper.debugShowObs(obs)

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

