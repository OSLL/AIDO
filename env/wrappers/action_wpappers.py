import gym
import numpy as np
from gym import spaces

from lane_control import Controller

class LeftRightBraking2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(LeftRightBraking2WheelVelsWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        return np.clip(np.array([1., 1.]) - np.array(action), 0., 1.)


class LeftRightClipped2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(LeftRightClipped2WheelVelsWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        return np.clip(np.array(action), 0., 1.)


class Heading2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env, heading_type=None):
        super(Heading2WheelVelsWrapper, self).__init__(env)
        self.heading_type = heading_type
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        if self.heading_type == 'heading_trapz':
            straight_plateau_half_width = 0.3333  # equal interval for left, right turning and straight
            self.mul = 1. / (1. - straight_plateau_half_width)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        # action = [-0.5 * action + 0.5, 0.5 * action + 0.5]
        if self.heading_type == 'heading_smooth':
            action = np.clip(np.array([1 + action ** 3, 1 - action ** 3]), 0., 1.)  # Full speed single value control
        elif self.heading_type == 'heading_trapz':
            action = np.clip(np.array([1 - action, 1 + action]) * self.mul, 0., 1.)
        elif self.heading_type == 'heading_sine':
            action = np.clip([1 - np.sin(action * np.pi), 1 + np.sin(action * np.pi)], 0., 1.)
        elif self.heading_type == 'heading_limited':
            action = np.clip(np.array([1 + action * 0.666666, 1 - action * 0.666666]), 0., 1.)
        else:
            action = np.clip(np.array([1 + action, 1 - action]), 0., 1.)  # Full speed single value control
        return action


class SteeringBraking2WheelVelsWrapper(gym.ActionWrapper):
    """
    Input: action vector
        action[0] - steering
        action[1] - braking
    Output: action vector:
        wheel velocities
    """

    def __init__(self, env, heading_type=None):
        super(SteeringBraking2WheelVelsWrapper, self).__init__(env)
        self.heading_type = heading_type
        self.action_space = spaces.Box(low=np.array([-1., 0.]), high=np.array([1., 1.]))

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        action = np.clip(np.array([1 + action[0], 1 - action[0]]), 0., 1.)  # Full speed single value control
        action *= np.clip(1. - action[1], 0., 1.)
        return action


class ActionSmoothingWrapper(gym.ActionWrapper):
    def __init__(self, env, ):
        super(ActionSmoothingWrapper, self).__init__(env)
        self.last_action = np.array([0.])
        self.new_action_ratio = 0.75

    def action(self, action):
        smooth_action = (1. - self.new_action_ratio) * self.last_action + self.new_action_ratio * action[0]
        self.last_action = action[0]
        return (smooth_action)

    def reset(self, **kwargs):
        self.last_action = np.zeros(self.action_space.shape)
        return self.env.reset(**kwargs)


'''class PIDController(gym.ActionWrapper):
    def __init__(self, env, data=(1, 0.1, 0)):
        super(PIDController, self).__init__(env)
        p, i, d = data
        self.pid = PID(p, i, d)
        self.pid.output_limits = (-0.636, 0.636)
        self.pid.sample_time = 0.001
        self.r = 0.0318
        self.gain = 1
        self.base_line = 0.1

    def vel_to_omega(self, action):
        v1 = action[0][0]
        v2 = action[1][0]
        v = (self.r * v1 + self.r * v2) / 2.0
        omega = (self.r * v1 - self.r * v2) / self.base_line
        print(f'before {v2, v2}')
        return [v, omega]

    def omega_to_vel(self, v, omega):
        v1 = (v + 0.5 * omega * self.base_line) / self.r
        v2 = (v - 0.5 * omega * self.base_line) / self.r
        print(f'after {v1, v2}')
        return [[v1], [v2]]

    def action(self, action):
        action = self.vel_to_omega(action)
        omega = self.pid(action[1])
        print(f'omega {omega}')
        return self.omega_to_vel(action[0], omega)
'''
class PIDAction(gym.ActionWrapper):
    def __init__(self, env):
        super(PIDAction, self).__init__(env)
        self.controller = Controller()

    def action(self, action):
        #print(2)
        #data = self.unwrapped.get_lane_pos2(self.unwrapped.cur_pos, self.unwrapped.cur_angle)
        #action = self.controller.compute_action((data.dist, data.angle_deg))
        action = self.controller.compute_action((action[0], action[1]))
        return action

