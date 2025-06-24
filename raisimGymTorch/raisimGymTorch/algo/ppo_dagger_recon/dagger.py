import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# “.” means the module storage is in the same directory as the importing script
from .storage import ObsStorage 
import copy


# computes and returns the latent from the expert 【RETURN：Encoder output】
class DaggerExpert(nn.Module):
    def __init__(self, path, total_obs_size, T, tobeEncode_dim, prop_latent_dim, nenvs,actor_expert, critic_r, device):
        super(DaggerExpert, self).__init__()

        self.policy = torch.load(path,map_location={'cpu': device})
        self.actor_expert = actor_expert
        self.critic_r = critic_r

        self.actor_expert.architecture.load_state_dict(self.policy['actor_architecture_state_dict'])
        self.actor_expert.distribution.load_state_dict(self.policy['actor_distribution_state_dict'])
        self.critic_r.architecture.load_state_dict(self.policy['critic_r_architecture_state_dict'])

        self.T = T
        self.total_obs_size = total_obs_size
        self.tobeEncode_dim = tobeEncode_dim

        loadpth = path.split('full')[0]
        runid = path.split('full')[1].split('.pt')[0]
        mean_pth = loadpth + "/mean" + runid + ".csv"
        var_pth = loadpth + "/var" + runid + ".csv"
        obs_mean = np.loadtxt(mean_pth, dtype=np.float32)
        obs_var = np.loadtxt(var_pth, dtype=np.float32)
        self.mean = self.get_tiled_scales(obs_mean, nenvs, total_obs_size, tobeEncode_dim, T)
        self.var = self.get_tiled_scales(obs_var, nenvs, total_obs_size, tobeEncode_dim, T)

        self.prop_latent_dim = prop_latent_dim

    def get_tiled_scales(self, invec, nenvs, total_obs_size, tobeEncode_dim, T):
        outvec = np.zeros([nenvs, total_obs_size], dtype = np.float32)

        # ly NO dagger
        outvec[:, :tobeEncode_dim * (T+1)] = np.tile(invec[0, :tobeEncode_dim], [1, (T+1)])
        outvec[:, tobeEncode_dim * (T+1):] = invec[0, tobeEncode_dim:]
        return outvec

    def forward(self, obs):
        # obs = obs[:,-self.tail_size:]
        expert_latent = obs[:,(self.T + 1) * self.tobeEncode_dim : (self.T + 1) * self.tobeEncode_dim + self.prop_latent_dim]
        return expert_latent





class DaggerAgent:
    def __init__(self, expert_policy,
                 prop_latent_encoder,
                 T, tobeEncode_dim, device):

        expert_policy.to(device)
        prop_latent_encoder.to(device)
        self.expert_policy = expert_policy
        self.prop_latent_encoder = prop_latent_encoder.to(device)
        self.tobeEncode_dim = tobeEncode_dim

        self.T = T
        self.device = device
        self.mean = expert_policy.mean
        self.var = expert_policy.var
        self.student_actor = copy.deepcopy(self.expert_policy.actor_expert)

    def get_history_encoding(self, obs):
        hlen = self.tobeEncode_dim * self.T
        raw_obs = obs[:, : hlen]
        prop_latent = self.prop_latent_encoder(raw_obs)
        return prop_latent

    def get_expert_action(self, obs):
        hlen = self.tobeEncode_dim * self.T
        teacher_obs = obs[:, hlen + (self.tobeEncode_dim) :]
        with torch.no_grad():
            output = self.expert_policy.actor_expert.architecture.architecture(teacher_obs)
        return output

    def get_student_action(self, obs):
        hlen = self.tobeEncode_dim * self.T
        prop_latent = self.get_history_encoding(obs)
        student_obs = torch.cat([obs[:, hlen: hlen + self.tobeEncode_dim], prop_latent,
                            obs[:, hlen + self.tobeEncode_dim + prop_latent.shape[1]:]], 1)
        with torch.no_grad():
            output = self.student_actor(student_obs)
        return output

    def get_expert_latent(self, obs):
        latent = self.expert_policy(obs).detach()
        return latent

    def save_deterministic_graph(self, fname_prop_encoder,
                                 fname_mlp, device='cpu'):
        prop_encoder_graph = torch.jit.script(self.prop_latent_encoder.to(device))
        torch.jit.save(prop_encoder_graph, fname_prop_encoder)
        self.prop_latent_encoder.to(self.device)

class DaggerTrainer:
    def __init__(self,
            agent,
            num_envs, 
            num_transitions_per_env,
            obs_shape, latent_shape,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4,
            update_mlp=False):

        self.agent = agent
        self.storage = ObsStorage(num_envs, num_transitions_per_env, [obs_shape], [latent_shape], device)
        self.update_mlp = update_mlp
        if self.update_mlp:
            self.optimizer = optim.Adam([*self.agent.prop_latent_encoder.parameters(), *self.agent.student_actor.parameters()],
                                    lr=learning_rate)
        else:
            self.optimizer = optim.Adam([*self.agent.prop_latent_encoder.parameters()],
                                    lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    def observe_student(self, obs):
        actions = self.agent.get_student_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def observe_teacher(self, obs):
        actions = self.agent.get_expert_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def step(self, obs):
        expert_latent = self.agent.get_expert_latent(torch.from_numpy(obs).to(self.device))
        self.storage.add_obs(obs, expert_latent)

    def update(self):
        # Learning step
        avg_prop_loss, avg_action_loss = self._train_step()
        self.storage.clear()
        return avg_prop_loss, avg_action_loss

    def _train_step(self):
        for epoch in range(self.num_learning_epochs):
            # return loss in the last epoch
            prop_mse = 0
            action_mse = 0
            loss_counter = 0
            for obs_batch, expert_action_batch in self.storage.mini_batch_generator_inorder(self.num_mini_batches):

                predicted_prop_latent = self.agent.get_history_encoding(obs_batch)

                actions_expert = self.agent.get_expert_action(obs_batch)
                actions_predict = self.agent.get_student_action(obs_batch)

                loss_action = self.loss_fn(actions_predict, actions_expert)
                loss_prop = self.loss_fn(predicted_prop_latent, expert_action_batch)
                loss = loss_prop + loss_action

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                prop_mse += loss_prop.item()
                action_mse += loss_action.item()
                loss_counter += 1

            avg_prop_loss = prop_mse / loss_counter
            avg_action_loss = action_mse / loss_counter

        self.scheduler.step()
        return avg_prop_loss, avg_action_loss