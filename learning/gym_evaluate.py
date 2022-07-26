import torch
from ParamsNet import ParamsNet
from pyglet.window import key
import random
from env import Environment, ClipImageWrapper, ResizeWrapper, MotionBlurWrapper, NormalizeWrapper, PIDAction

env_config = {
    "seed": random.randint(0, 100000),
    "map_name": "ETU_autolab_track",
    "max_steps": 1000,
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
    model = ParamsNet()
    model.load('./model/90000')
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