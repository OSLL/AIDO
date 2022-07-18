import pyglet
import torch
from pyglet.window import key
from ParamsNet import ParamsNet
import random
from env import Environment, ClipImageWrapper, ResizeWrapper, MotionBlurWrapper, NormalizeWrapper, PIDAction
import numpy as np

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


def warp(env):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)
    env = PIDAction(env)
    return env

if __name__ == '__main__':
    env = Environment(12345).create_env(False, env_config, warp)
    model = ParamsNet()
    model.load('./model/9250')
    model.eval()
    with torch.no_grad():
        env.reset()
        for i in range(100):
            obs = env.reset()
            done = False
            while not done:
                val = model.forward(obs)
                print(val)
                action = [val[0][0].item(), val[0][1].item()]
                print(action)
                obs, rew, done, info = env.step(action)
                env.render()