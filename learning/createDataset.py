from env import Environment, ClipImageWrapper, MotionBlurWrapper, ResizeWrapper, NormalizeWrapper, DatasetWrapper
import random

env_config = {
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


def warp(env, file, directory):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)
    env = DatasetWrapper(env, file, directory)
    return env

if __name__ == "__main__":
    file = open('./dataset/dataset.csv', "w+")
    directory = './dataset'
    env = Environment(123456).dataset(env_config=env_config, warp=warp, file=file, directory=directory)
    env.reset()
    done = False
    for i in range(1000000):
        if done:
            env.reset()
        action = [random.uniform(0, 1), random.uniform(0, 1)]
        obs, rew, done, info = env.step(action)
    file.close()
