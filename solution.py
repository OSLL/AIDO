#!/usr/bin/env python3
import os
from typing import Optional
from learning import ParamsNet
import numpy as np
from lane_control import Controller

from aido_schemas import EpisodeStart, protocol_agent_DB20, PWMCommands, DB20Commands, LEDSCommands, RGB, \
    wrap_direct, Context, DB20Observations, JPGImage, logger

from env import Environment, ClipImageWrapper, ResizeWrapper, NormalizeWrapper, get_wrappers
from PIL import Image

import io
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


def warp(env):
    env = ClipImageWrapper(env, 3)
    env = ResizeWrapper(env, (64, 64))
    env = NormalizeWrapper(env)
    return env


class PytorchRLTemplateAgent:
    def __init__(self, load_model: bool, model_path: Optional[str]):
        self.action_counter = 0
        self.obs_counter = 0
        self.i = 0
        self.load_model = load_model
        self.model_path = model_path
        self.controller = Controller()

        ### action parts ###
        self.last_action = np.array([0.])
        self.new_action_ratio = 0.75

    def init(self, context: Context):
        self.check_gpu_available(context)
        logger.info('PytorchRLTemplateAgent init')
        self.model = ParamsNet()
        self.model.load('./model/90000')
        self.env = Environment(123345).wrap(env_config, warp)
        self.obs_wrappers, _, _ = get_wrappers(self.env)
        self.current_image = np.zeros((640, 480, 3))
        logger.info('PytorchRLTemplateAgent init complete')

    def check_gpu_available(self, context: Context):
        import torch
        available = torch.cuda.is_available()
        req = os.environ.get('AIDO_REQUIRE_GPU', None)
        context.info(f'torch.cuda.is_available = {available!r} AIDO_REQUIRE_GPU = {req!r}')
        context.info('init()')
        if available:
            i = torch.cuda.current_device()
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(i)
            context.info(f'device {i} of {count}; name = {name!r}')
        else:
            if req is not None:
                msg = 'I need a GPU; bailing.'
                context.error(msg)
                raise RuntimeError(msg)

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: DB20Observations):
        logger.info(f"OBS: {self.obs_counter}")
        self.obs_counter += 1
        count = self.obs_counter - 1
        camera: JPGImage = data.camera
        obs = jpg2rgb(camera.jpg_data)
        # ----------------------------------------
        for idx, obs_wrap in enumerate(self.obs_wrappers):
            # print(f"counter: {idx}; type:{type(obs_wrap)}; img: {type(self.current_image)}, {self.current_image.shape}")
            logger.info(f"counter: {idx}; type:{type(obs_wrap)}; img: {type(obs)}, {obs.shape}")
            obs = obs_wrap.observation(obs)

        self.current_image = obs
        logger.info(f"OBS ENDD: {count}")
        # ----------------------------------------------
        # self.current_image = self.obs_wrappers[-1].observation(self.current_image)
        # print(f"[After wrappers]: {type(self.current_image)}, {self.current_image.shape}")
        # print(type(obs))
        # print(obs.shape)

    def compute_action(self, observation):
        self.i += 1

        val = self.model.forward(observation)
        action = [val[0][0].item(), val[0][1].item()]
        return self.controller.compute_action((action[0], action[1]))


    def on_received_get_commands(self, context: Context):
        # self.action_counter
        logger.info(f"Action: {self.action_counter}")
        self.action_counter += 1
        pwm_left, pwm_right = self.compute_action(self.current_image)

        pwm_left = float(np.clip(pwm_left, 0, +1))
        pwm_right = float(np.clip(pwm_right, 0, +1))

        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = PytorchRLTemplateAgent(load_model=False, model_path=None)
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
