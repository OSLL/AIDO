import torch
from ParamsNet import ParamsNet
from pyglet.window import key
import random
import time
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
    model = ParamsNet(2)
    #phi = ParamsNet(1)
    #phi.load('./phi/50000')
    #phi.eval()
    #model.load('./phi_deg/10000')
    #model.load('./model/90000')
    model.load('./model_with_real/20000')
    model.eval()
    with torch.no_grad():
        env.reset()
        for i in range(100):
            obs = env.reset()
            done = False
            while not done:
                t =time.time()

                val = model.forward(obs)
                #v = phi.forward(obs)
                print(f'time = {time.time() - t}')
                print(val)
                #action = [val[0][0].item(), v[0].item()]
                action = [val[0][0].item(), val[0][1].item()]
                print(action)
                obs, rew, done, info = env.step(action)

                env.render()