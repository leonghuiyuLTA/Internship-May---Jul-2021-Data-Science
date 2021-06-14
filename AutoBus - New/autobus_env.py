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
        self.reward_weights = [1.5,1,  500]# [2,1,10,1000,1000] 5,, 10000, 50001,
        self.maxtime = 50

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
        if self.time > self.maxtime: self.done = True
        self.jerk = abs((action - self.prev_acceleration))
        self.prev_acceleration = action

        reward_list = self.get_reward()
        self.position = self.new_position
        info = {
            'distance left' : self.position,
            'velocity' : self.velocity * 3.6,
            'acceleration': self.prev_acceleration,
            'jerk': self.jerk,
            'time': self.time,
        }
        state = self.get_state()
        reward = np.array(reward_list).dot(np.array(self.reward_weights))
        return state, reward, self.done, info

    def get_state(self):
        return np.hstack((self.position, self.velocity * 3.6, self.prev_acceleration))

    def get_reward(self):
        # if exceed speed limit, higher penalty. Else, penalty is the difference between self velocity and speed limit
        if self.position > 0:
            if self.velocity <= self.speed_limit:
                reward_forward = (self.velocity - self.speed_limit)/self.speed_limit
            else:
                reward_forward = -(self.velocity - self.speed_limit)*50
        else:
            reward_forward = 0
        # penalty for jerk
        reward_jerk = - self.jerk
        # penalty for velocity >0 at/after end point
        reward_vel = -self.velocity**8 if self.position <= 0 else 0
        reward_acc = -5 if self.velocity == 0 and self.prev_acceleration < 0 else 0
        # penalty for not reaching end position/ penalty for exceeding end position
        if self.time == self.maxtime and self.new_position > 0:
            reward_pos = -(self.new_position)**2
        # elif self.new_position<0:
        #     reward_pos = self.new_position**5
        else:
            reward_pos = 0
        reward_list = [
            reward_forward, reward_jerk,  reward_vel#,, reward_posreward_acc,
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

