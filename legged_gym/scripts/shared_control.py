# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.policy_utils import ppo_inference_torch, control_neuron_activation
import numpy as np
import torch


def play(args, map, activation_func="elu", model_name=None):
    assert activation_func == "elu" or activation_func == "tanh", "only support elu or tanh"
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

    name = model_name or ("anymal" if "anymal" in args.task else "cassie")
    policy_weights = np.load("{}_{}.npz".format(name, activation_func))

    for i in range(10 * int(env.max_episode_length)):
        command = "Forward"
        # if command == "Forward":
        #     print(i)

        obs[..., 9:12] = torch.Tensor([0. , 0., .0])
        actions, _ = ppo_inference_torch(policy_weights, obs.clone().cpu().numpy(),
                                         map,
                                         command,
                                         activation=activation_func,
                                         deterministic=True)
        actions = torch.unsqueeze(torch.from_numpy(actions.astype(np.float32)), dim=0)
        obs, _, rews, dones, infos = env.step(actions)


if __name__ == '__main__':
    cassie_elu = {"Right": {1: [(169, 10)]},
                  "Forward": {2: [(2, 5)]},
                  "z": {0: [(267, 7), (497, 10)]}
                  }
    z_0_cassie_elu = {"Right": {1: [(169, 10)]},
                      "Forward": {1: [(40, 28)], 0:[(487, -10)]},
                      "z": {0: [(394, 146), (170, 143), (487, -100)]}
                      }
    anymal_elu = {"Forward": {0: [(143, -15)]}}
    args = get_args()
    args.num_envs = 1

    # args.task = "anymal_c_rough"
    args.task = "cassie"
    activation = "elu"
    play(args, activation_func=activation, map=z_0_cassie_elu, model_name="0_z_cassie")
