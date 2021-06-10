import pygame
import rendering
import numpy as np
from gym import spaces
from gym.utils import seeding
# VERSION 1 : BUS SUPPOSED TO DRIVE AT MAX SPEED UNTIL REACH GOAL - 200m


class AutobusEnv:

    def __init__(self):
        self.speed_limit = 50/3.6
        self.dt = 0.25
        self.reward_weights = [1.0]

        self.dist_to_bus_stop = 200.0
        self.position = self.dist_to_bus_stop
        self.new_position = self.position
        self.velocity = 0.0
        self.prev_acceleration = 0.0
        self.jerk = 0.0  # new acc - old acc
        self.time = 0
        self.done = False

        self.action_space = spaces.Box(low = -4, high = 4, shape = (1,))

        self.viewer = None

    def step(self, action):
        # TODO: Allow for exceeding of bus stop
        if self.position - (0.5 * action * self.dt ** 2 + self.velocity * self.dt) > self.position:
            self.new_position = self.position
        else:
            self.new_position = self.position - (0.5 * action * self.dt ** 2 + self.velocity * self.dt)
        if self.new_position <= 0 and self.velocity == 0:
            self.done = True
        if self.velocity + action * self.dt >= 0:
            self.velocity += action * self.dt
        else:
            self.velocity= 0
        self.time += self.dt
        if self.time > 300: self.done = True
        self.jerk = abs((action - self.prev_acceleration))
        self.prev_acceleration = action

        reward_list = self.get_reward()
        self.position = self.new_position
        info = {
            'distance left' : self.position,
            'velocity' : self.velocity,
            'acceleration': self.prev_acceleration,
            'jerk': self.jerk,
            'time': self.time,
        }
        state = self.get_state()
        reward = np.array(reward_list).dot(np.array(self.reward_weights))
        return state, reward, self.done, info

    def get_state(self):
        return np.hstack((self.position, self.velocity, self.prev_acceleration))

    def get_reward(self):
        # TODO: to generate reward functions here - in terms of velocity.
        reward_list = [

        ]
        return reward_list

    def reset(self):
        self.position = self.dist_to_bus_stop
        self.new_position = self.position
        self.velocity = 0.0
        self.prev_acceleration = 0.0
        self.jerk = 0.0  # new acc - old acc
        self.time = 0
        self.done = False
        state = self.get_state()
        return state

    def render(self):
        rendering.BusViewer().update_screen(self.dist_to_bus_stop - self.position, self.velocity, self.time)

    def close(self):
        rendering.BusViewer().close()

