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
        self.reward_weights = [1.0, 5.0/(self.speed_limit**2), 0.1]# [2,1,10,1000,1000] 5,, 10000, 50001,

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
        self.jerk = abs((action - self.prev_acceleration))/self.dt
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
        reward += self.get_terminal_reward()
        return state, reward, self.done, info

    def get_state(self):
        return np.hstack((self.position, self.velocity * 3.6, self.prev_acceleration))

    def get_reward(self):
        pos_reward = 0.0
        vel_reward = 0.0 
        acc_reward = 0.0 
        jerk_reward =0.0 
        
        # if exceed speed limit, higher penalty. Else, penalty is the difference between self velocity and speed limit
        if self.position < 0.0: 
            pos_reward = -self.velocity**2
        else: 
            vel_reward = self.velocity**2
            if(self.velocity > self.speed_limit): 
                vel_reward -= (self.velocity - self.speed_limit)**2
                
        jerk_reward = -abs(self.jerk)
            
            
        # penalty for jerk
        # penalty for velocity >0 at/after end point
        reward_acc = -5 if self.velocity == 0 and self.prev_acceleration < 0 else 0

        reward_list = [
            pos_reward, vel_reward,  jerk_reward
            #,, reward_posreward_acc,
        ]
        return reward_list
    
    def get_terminal_reward(self): 
        if(self.time == self.maxtime): 
            return -self.new_position**2
        else:
            if(self.position < 0.0): 
                return -self.position**2
            return 0.0 

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

