import os
import shutil
import glob
import time

try:
    from pyglet.window import key
except Exception:
    pass

from datetime import datetime
import csv
import torch
import numpy as np

import gym


from PPO import PPO
from env import DtRewardTargetOrientation, ClipImageWrapper, ResizeWrapper, MotionBlurWrapper, PIDController, \
    NormalizeWrapper, Heading2WheelVelsWrapper, DtRewardPosingLaneWrapper, DtRewardVelocity, DtRewardCollisionAvoidance, PIDAction, DtRewardWrapperDAndPhi
import gym
from env import Environment
from PPO import PPO
import random

env_config = {
    "seed": random.randint(0, 100000),
    "map_name": "loop_empty",
    "max_steps": 5000,
    "camera_width": 640,
    "camera_height": 480,
    "accept_start_angle_deg": 40,
    "full_transparency": True,
    "distortion": True,
    "domain_rand": False
}

def warp1(env):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)
    #env = PIDAction(env)
    env = DtRewardWrapperDAndPhi(env)
    return env


def warp(env):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    #env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)
    #env = PIDController(env, data)
    #env = Heading2WheelVelsWrapper(env)
    #env = DtRewardTargetOrientation(env)
    #env = DtRewardVelocity(env)
    #env = DtRewardCollisionAvoidance(env)
    #env = DtRewardPosingLaneWrapper(env)
    return env


#################################### Testing ###################################
def test(evaluate=False):
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "Duckietown"
    has_continuous_action_space = True
    max_ep_len = 1000  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving

    render = True  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 10  # total num of testing episodes

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

    #####################################################
    env = None
    if evaluate:
        env = Environment(123456).create_env(env_config, warp1)
    else:
        env = Environment(123456).wrap(env_config, warp1)

    # state space dimension
    a, b, c = env.observation_space.shape
    state_dim = a * b * c

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # preTrained weights directory

    random_seed = 0  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained, -32)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)
    if evaluate:
        print("--------------------------------------------------------------------------------------------")

        test_running_reward = 0

        for ep in range(1, total_test_episodes + 1):
            ep_reward = 0
            state = env.reset()

            for t in range(1, max_ep_len + 1):
                state = np.array([state])
                action = ppo_agent.select_action(state)
                print(action)
                state, reward, done, _ = env.step(action)
                ep_reward += reward

                if render:
                    env.render()
                    time.sleep(frame_delay)

                if done:
                    break

            # clear buffer
            ppo_agent.buffer.clear()

            test_running_reward += ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            ep_reward = 0

        env.close()

        print("============================================================================================")

        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)
        print("average test reward : " + str(avg_test_reward))

        print("============================================================================================")

    return ppo_agent



if __name__ == '__main__':
    test(True)