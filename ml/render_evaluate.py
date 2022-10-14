import torch
from ml_model import ParallelNet, ParamsNet
from pyglet.window import key
import random
import time
import numpy as np
from env import Environment, ClipImageWrapper, ResizeWrapper, MotionBlurWrapper, NormalizeWrapper, PIDAction

env_config = {
    "seed": random.randint(0, 100000),
    "map_name": "loop_empty",
    "max_steps": 10000,
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
    env = Environment(12345).create_env(False, None, env_config, warp)
    #model = ParamsNet('cuda:0')
    model = ParallelNet('cuda:0')
    #model.load_state_dict(torch.load('../learning/best.pkl'))
    model.load_state_dict(torch.load('best.pkl'))
    model.to('cuda:0')
    model.eval()

    env.reset()
    for i in range(100):
        obs = env.reset()
        done = False
        while not done:
            t = time.time()
            obs = np.array([obs])
            bs, h, w, c = obs.shape
            obs = obs.reshape(bs, c, h, w)
            #val = model.forward(obs)
            d, phi = model.forward(obs)
            #action = [val[0][0].item(), val[0][1].item()]
            action = [d.item(), phi.item()]
            print(action)
            obs, rew, done, info = env.step(action)

            env.render()

