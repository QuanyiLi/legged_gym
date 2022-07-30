from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.policy_utils import ppo_inference_torch, control_neuron_activation
import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.mesh_type = "plane"
    train_cfg.runner.num_steps_per_env = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    env.set_camera((2, -5, 3), (0, 0, 0))

    policy_weights = np.load("cassie.npz")

    for i in range(10 * int(env.max_episode_length)):
        obs[..., 9:12] = torch.Tensor([.0, 0., 0.])
        actions, _ = ppo_inference_torch(policy_weights, obs.clone().cpu().numpy(), {"Forward": {0: [(79, 15)]}},
                                         "Forward",
                                         deterministic=False)
        actions = torch.unsqueeze(torch.from_numpy(actions.astype(np.float32)), dim=0)
        obs, _, rews, dones, infos = env.step(actions)


if __name__ == '__main__':
    args = get_args()
    args.num_envs = 1
    args.task = "cassie"
    play(args)
