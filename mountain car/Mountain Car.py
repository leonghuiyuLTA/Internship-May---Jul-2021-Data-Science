import numpy as np
from matplotlib import pyplot as plt
import gym

pos_bins = np.linspace(-1.2, 0.6, 10)
vel_bins = np.linspace(-0.07, 0.07, 10)

def get_state(obs):
    pos, vel = obs
    pos_state = np.digitize(pos, pos_bins)
    vel_state = np.digitize(vel, vel_bins)
    return (pos_state, vel_state)

def chooseaction(epsilon, Q, state, action = [0,1,2]):
    if np.random.random() < epsilon:
        return np.random.choice(action)

    values = np.array(Q[state,a] for a in action)
    return np.argmax(values)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    rounds = 0
    learn_rate = 0.1
    discount = 1
    epsilon = 1
    Q = {}
    states = []
    for p in range(11):
        for v in range(11):
            states.append((p,v))
    for state in states:
        for action in [0,1,2]:
            Q[state,action] = 0

    for i in range(rounds):
        done = False
        state = env.reset()
        action = chooseaction(epsilon, Q, state)
        newstate , reward, done, info = env.step(action)