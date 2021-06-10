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
        # V1: [1.0]
        # V2: [1.0, 1.0]
        # V3? : [1.4, 3.6, 2.6, 4.0] and exceeding position * 8.5
        self.reward_weights = [1.4, 2.3, 1.4, 3.1] #, 2.0, 1.0]  # , 1.0, 10.0, 2.0]
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
        # Calculate the distance remaining to the bus stop
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

        reward_list = self.get_reward() # is a list, but now we only care speed limits
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
        # reward for distance remaining to the bus stop
        if self.new_position >= 0:
            reward_distance = (self.position - self.new_position)
        elif self.position >= 0:
            reward_distance = (self.position + self.new_position) * 20  # <- problem here
        else:
            # V1: = 0
            # V2: = self.position
            reward_distance = self.new_position * 20
        # else:
        #     reward_distance = self.new_position

        # reward for not jerking
        reward_jerk = -self.jerk/5 # 10 is max jerk

        # penalty if exceed speed limit
        reward_vel = self.speed_limit - self.velocity if self.velocity > self.speed_limit else 0
        reward_vel = reward_vel - self.velocity ** 2 if self.position <= 0 else reward_vel

        # reward for shorter time taken to reach the end
        reward_time = - self.dt

        # V1: reward_distance
        # V2: reward_distance, reward_time
        reward_list = [
            reward_distance, reward_time, reward_jerk, reward_vel
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

