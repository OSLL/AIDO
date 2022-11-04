import random
from env import Environment, ClipImageWrapper, ResizeWrapper, MotionBlurWrapper, NormalizeWrapper, BatchWrapper, DandPhiWrapper
import numpy as np



def warp(env):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)
    env = DandPhiWrapper(env)
    env = BatchWrapper(env)
    return env


class BatchGenerator:
    def __init__(self, size):
        self.env__ = Environment(122345).create_env(map='ETU_autolab_track')
        self.env_ = Environment(122345).create_env(map='loop_empty')
        self.env3 = Environment(123457).create_env(wrap=warp)
        self.env4 = Environment(123457).create_env(wrap=warp, map="loop_empty")
        self.size = size
        self.obs_batch = []
        self.d_batch = []
        self.phi_batch = []
        self.env_list = [self.env_, self.env__, self.env3, self.env4]

    def create_batch(self):
        current_size = 0
        del self.obs_batch, self.d_batch, self.phi_batch
        self.obs_batch = []
        self.d_batch = []
        self.phi_batch = []
        while current_size < self.size:
            print(self.env_list)
            for env in self.env_list:
                obs = env.reset()
                done = False
                if done:
                    obs = env.reset()
                obs_, rew, done, info = env.step([random.uniform(-1, 1), random.uniform(-1, 1)])
                if info["successfully"]:
                    self.obs_batch.append(obs)
                    self.d_batch.append(info['dist'])
                    self.phi_batch.append(info['rad'])
                    current_size += 1
                obs = obs_
        return np.array(self.obs_batch), np.array(self.d_batch), np.array(self.phi_batch)
