try:
    import pyglet

    pyglet.window.Window()
except Exception:
    pass
import random
import numpy as np
try:
    from gym_duckietown.simulator import Simulator, DEFAULT_ROBOT_SPEED, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
    from gym_duckietown.envs import DuckietownEnv
except:
    pass
from .wrappers.general_wrappers import InconvenientSpawnFixingWrapper, DummyDuckietownGymLikeEnv
from .wrappers.observe_wrappers import ResizeWrapper, NormalizeWrapper, ClipImageWrapper, MotionBlurWrapper
from .wrappers.envWrapper import BatchWrapper


class Environment:
    def __init__(self, seed):
        self._env = None
        np.random.seed(seed)
        random.seed(seed)
        self.default = {
            "seed": random.randint(0, 100000),
            "map_name": "ETU_autolab_track",
            "max_steps": 5000,
            "camera_width": 640,
            "camera_height": 480,
            "accept_start_angle_deg": 40,
            "full_transparency": True,
            "distortion": True,
            "domain_rand": False
        }

    def create_env(self, default=True, map=None, env_config=None, wrap=None):
        self._env = None
        if default:
            if map is None:
                env_config = self.default
            else:
                env_config = self.default
                env_config["map_name"] = map
            if wrap is None:
                wrap = self.default_warp
        print(default)
        print(env_config)

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

    def default_warp(self, env):
        print(type(env))
        env = ClipImageWrapper(env, 3)
        env = ResizeWrapper(env, (64, 64))
        env = MotionBlurWrapper(env)
        env = NormalizeWrapper(env)
        env = BatchWrapper(env)
        return env

