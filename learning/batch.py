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


class Batch:
    def __init__(self, size, type_):
        self.env = None
        self.env__ = Environment(122345).create_env()
        self.env_ = Environment(122345).create_env(map="loop_empty")
        self.env3 = Environment(123457).create_env(wrap=warp)
        self.env4 = Environment(123457).create_env(wrap=warp, map="loop_empty")
        self.size = size
        self.obs_batch = []
        self.res_batch = []
        self.type = type_
        self.index = 0

    def create_batch(self):
        if self.index == 0:
            self.env = self.env_
            self.index = 1
        elif self.index == 1:
            self.env = self.env__
            self.index = 2
        elif self.index == 2:
            self.env = self.env3
            self.index = 3
        elif self.index == 3:
            self.env = self.env4
            self.index = 0
        obs = self.env.reset()
        done = False
        del self.obs_batch, self.res_batch
        self.obs_batch = []
        self.res_batch = []
        current_size = 0
        while current_size < self.size:
            if done:
                obs = self.env.reset()
            obs_, rew, done, info = self.env.step([random.uniform(-1, 1), random.uniform(-1, 1)])
            if info["successfully"]:
                self.obs_batch.append(obs)
                tmp =  [info['dist'], info['rad']]
                self.res_batch.append(tmp)
                current_size += 1
            obs = obs_
        return np.array(self.obs_batch), np.array(self.res_batch)
