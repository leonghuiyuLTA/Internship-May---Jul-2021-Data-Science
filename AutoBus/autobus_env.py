import pygame
import rendering
import numpy as np
from gym import spaces
from gym.utils import seeding
# VERSION 1 : BUS SUPPOSED TO DRIVE AT MAX SPEED UNTIL REACH GOAL - 200m

class AutobusEnv():
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.speed_limit = 50
        self.track_length = 3000
        self.dt = 0.1

        self.reward_weights = [1.0, 1.0] # 1 for reaching curr speed limit, 1.0 for jerk?
        self.seed()

        self.position = 0.0
        self.velocity = 0.0
        self.prev_acceleration = 0.0
        self.jerk = 0.0 # new acc - old acc
        self.time = 0
        self.done = False

        self.action_space = spaces.Box(low = -4, high = 4, shape = (1,))

        self.viewer = None

    def step(self,action):
        # assert self.action_space.contains(action), f'{action} ({type(action)}) invalid shape or bounds'
        self.position += (0.5 * action * self.dt ** 2 + self.velocity * self.dt)
        self.velocity += action * self.dt
        self.time += self.dt
        self.jerk = abs((action - self.prev_acceleration))
        self.prev_acceleration = action

        self.done = bool(self.position >= self.track_length)
        reward_list = self.get_reward() # is a list, but now we only care speed limits
        info = {
            'position' : self.position,
            'velocity' : self.velocity,
            'acceleration': self.prev_acceleration,
            'jerk': self.jerk,
            'time': self.time,
        }
        state = self.get_state()
        reward = np.array(reward_list).dot(np.array(self.reward_weights))
        return state, reward, self.done, info

    def get_state(self):
        return np.hstack((self.velocity, self.prev_acceleration))

    def get_reward(self):
        reward_forward = abs(self.velocity - self.speed_limit)
        reward_forward /= self.speed_limit

        reward_jerk = self.jerk / 5      # 5 is random number
        reward_list = [
            -reward_forward, -reward_jerk
        ]
        return reward_list

    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        self.prev_acceleration = 0.0
        self.jerk = 0.0  # new acc - old acc
        self.time = 0
        self.done = False
        state = self.get_state()
        return state

    def render(self):
        rendering.BusViewer().update_screen(self.position, self.velocity)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]