import argparse
from itertools import count
import os
import time

import gym
import scipy.optimize
import numpy as np

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from gif_recorder import save_episode_gif, plot_rewards, setup_gif_recording

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v4", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-gif-interval', type=int, default=100, metavar='N',
                    help='interval between saving GIFs (default: 100)')
parser.add_argument('--gif-duration', type=int, default=5, metavar='N',
                    help='duration of GIF in seconds (default: 5)')
parser.add_argument('--output-dir', type=str, default='gifs',
                    help='directory to save GIFs (default: gifs)')
parser.add_argument('--plot-interval', type=int, default=50, metavar='N',
                    help='interval between plotting rewards (default: 50)')
args = parser.parse_args()

env = gym.make(args.env_name, render_mode='rgb_array')

# Mujoco環境のレンダリング設定
if hasattr(env, 'env') and hasattr(env.env, 'model'):
    # Mujoco環境の場合、レンダリング設定を調整
    try:
        env.env.model.vis.global_.offwidth = 640
        env.env.model.vis.global_.offheight = 480
    except:
        pass

# 環境のレンダリングモードを設定
try:
    # 一部の環境では、レンダリングモードを明示的に設定する必要がある
    if hasattr(env, 'render_mode'):
        env.render_mode = 'rgb_array'
    elif hasattr(env, 'env') and hasattr(env.env, 'render_mode'):
        env.env.render_mode = 'rgb_array'
except:
    pass

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

# GIF保存用のディレクトリを設定
setup_gif_recording(args.output_dir)

# 報酬履歴を保存するリスト
reward_history = []
episode_history = []

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        reset_result = env.reset(seed=args.seed)
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result
        state = running_state(state)

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            step_result = env.step(action)
            
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
                episode_done = done or truncated
            else:
                next_state, reward, done, _ = step_result
                episode_done = done
                
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if episode_done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if episode_done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()
    update_params(batch)

    # 報酬履歴を保存
    reward_history.append(reward_batch)
    episode_history.append(i_episode)

    # 定期的にGIFを保存
    if i_episode % args.save_gif_interval == 0:
        save_episode_gif(env, policy_net, running_state, i_episode, reward_batch, 
                        args.output_dir, args.env_name, args.seed, select_action, args.gif_duration)

    # 定期的に報酬プロットを更新
    if i_episode % args.plot_interval == 0:
        plot_rewards(episode_history, reward_history, args.output_dir)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
