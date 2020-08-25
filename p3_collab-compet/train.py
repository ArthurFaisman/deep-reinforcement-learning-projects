# based on : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py

# Reference: https://arxiv.org/pdf/1509.02971.pdf
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from unityagents import UnityEnvironment

import argparse
from distutils.util import strtobool
import numpy as np
import time
import random
import os

from helpers import ReplayBuffer, Critic, Actor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG agent')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanrlddpg",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--wandb-api-key', type=str, default=None,
                        help="the wandb API key")

    args = parser.parse_args()

args.gym_id = "MLAgents-Tennis"
args.buffer_size = int(10000)
args.gamma = 0.99
args.tau = 1e-2
args.max_grad_norm = 0.5
args.batch_size = 128
args.exploration_noise = 0.5 #0.1
args.learning_starts = args.buffer_size
args.policy_frequency = 5
args.noise_clip = 0.5
args.total_episodes = 10000000
args.learning_rate = 1e-4

experiment_name = f"{args.gym_id}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    if args.wandb_api_key is not None:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key

    import wandb

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
               name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True

uenv = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis")
brain_name = uenv.brain_names[0]
brain = uenv.brains[brain_name]
# reset the environment
env_info = uenv.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size # number of actions
state_size = env_info.vector_observations.shape[1] # number of observations/states

num_agents = len(env_info.agents)
action_min = -1.0
action_max = 1.0


rb = ReplayBuffer(args.buffer_size)
actor = Actor(device, action_size, state_size).to(device)
critic = Critic(device, action_size, state_size).to(device)
critic_target = Critic(device, action_size, state_size).to(device)
target_actor = Actor(device, action_size, state_size).to(device)
target_actor.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())
critic_optimizer = optim.Adam(list(critic.parameters()), lr=args.learning_rate)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()


env_info = uenv.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
scores = np.zeros(num_agents)
episode_num = 0
global_step = 0
scores_window = deque(maxlen=100)  # last 100 scores

print("starting..")

while True:
    global_step += 1
    # ALGO LOGIC: put action logic here
    if global_step < args.learning_starts:
        actions = [action for action in np.random.randn(num_agents, action_size)]
    else:
        actions = [actor.forward(obs.reshape((1,state_size))) for obs in env_info.vector_observations]
        actions = [(action.tolist()[0]
                + np.random.normal(0, action_max * args.exploration_noise, size=action_size)
        ).clip(action_min, action_max) for action in actions]

    env_info = uenv.step(actions)[brain_name]
    next_states = env_info.vector_observations  # get next state (for each agent)
    rewards = env_info.rewards  # get reward (for each agent)
    dones = env_info.local_done

    scores += rewards

    # ALGO LOGIC: training.
    rb.put(zip(states, actions, rewards, next_states, dones))
    if global_step > args.learning_starts:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions = (
                target_actor.forward(s_next_obses)
            ).clamp(action_min, action_max)
            critic_next_target = critic_target.forward(s_next_obses, next_state_actions)
            next_critic_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args.gamma * (
                critic_next_target).view(-1)

        critic_a_values = critic.forward(s_obs, torch.Tensor(s_actions).to(device)).view(-1)
        critic_loss = loss_fn(critic_a_values, next_critic_value)

        # optimize the midel
        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(critic.parameters()), args.max_grad_norm)
        critic_optimizer.step()

        if global_step % args.policy_frequency == 0:
            actor_loss = -critic.forward(s_obs, actor.forward(s_obs)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    states = next_states

    if np.any(dones):
        episode_num += 1
        average_score = np.mean(scores)
        print(f"Total score (averaged over agents) for episode {episode_num} :\t {average_score}")
        writer.add_scalar("charts/episode_reward", average_score, episode_num)
        scores_window.append(average_score)
        obs, scores = uenv.reset(train_mode=True)[brain_name], np.zeros(num_agents)

        if episode_num >= args.total_episodes:
            break

        if np.mean(scores_window) >= 0.5:
            torch.save(actor.state_dict(), 'torch_model.save')
            break

uenv.close()
writer.close()
