try:
    import pyglet

    pyglet.window.Window()
except Exception:
    pass

import logging
import sys
import random
import numpy as np
try:
    from gym_duckietown.simulator import Simulator, DEFAULT_ROBOT_SPEED, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
    from gym_duckietown.envs import DuckietownEnv
except:
    pass
from .wrappers.general_wrappers import InconvenientSpawnFixingWrapper, DummyDuckietownGymLikeEnv
from .wrappers.observe_wrappers import ResizeWrapper, NormalizeWrapper, ClipImageWrapper, MotionBlurWrapper, RandomFrameRepeatingWrapper, ObservationBufferWrapper, RGB2GrayscaleWrapper, LastPictureObsWrapper, ReshapeWrapper
from .wrappers.reward_wrappers import DtRewardTargetOrientation, DtRewardVelocity, DtRewardCollisionAvoidance, DtRewardPosingLaneWrapper, DtRewardPosAngle
from .wrappers.action_wpappers import Heading2WheelVelsWrapper, ActionSmoothingWrapper
from .wrappers.envWrapper import ActionDelayWrapper, ForwardObstacleSpawnnigWrapper, ObstacleSpawningWrapper


class Environment:
    def __init__(self, seed):
        self._env = None
        np.random.seed(seed)
        random.seed(seed)

    def create_env(self, env_config, wrap):
        self._env = None
        try:
            self._env = Simulator(
                seed=env_config["seed"],
                map_name=env_config["map_name"],
                max_steps=env_config["max_steps"],
                camera_width=env_config["camera_width"],
                camera_height=env_config["camera_height"],
                accept_start_angle_deg=env_config["accept_start_angle_deg"],
                full_transparency=env_config["full_transparency"],
                distortion=env_config["distortion"],
                domain_rand=env_config["domain_rand"],
            )
        except:
            pass
        return wrap(self._env)

    def wrap(self, env_config, warp):
        env = DummyDuckietownGymLikeEnv()
        return warp(env)
