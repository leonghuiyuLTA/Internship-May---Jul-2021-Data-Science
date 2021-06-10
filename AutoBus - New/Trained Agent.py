import gc
import autobus_env
import numpy as np
import pickle
from matplotlib import pyplot as plt
import time

# set up the tiles (position, velocity, acceleration)
pos_intervals = np.linspace(0, 261, 261)
vel_intervals = np.linspace(0, 20, 80)
acc_intervals = np.linspace(-4, 5, 10)
action_space = np.arange(-4, 5, 1)


def get_state(obs):
    pos, vel, acc = obs
    pos_key = np.digitize(pos, pos_intervals)
    vel_key = np.digitize(vel, vel_intervals)
    acc_key = np.digitize(acc, acc_intervals)
    return pos_key, vel_key, acc_key


def choose_action(Q, state):
    max_index = 0
    max_val = -1e9
    for i in action_space:
        if Q[state, i] > max_val:
            max_val = Q[state, i]
            max_index = i
    return max_index


if __name__ == "__main__":
    env = autobus_env.AutobusEnv()
    pos_track = []
    vel_track = []
    acc_track = []
    time_track = []
    with open("policy", 'rb') as fo:
        Q = pickle.load(fo)
        fo.close()

    done = False
    score = 0
    init = env.reset()
    state = get_state(init)
    action = choose_action(Q, state)
    while not done:
        env.render()
        obs, reward, done, info = env.step(action)
        state = get_state(obs)
        action = choose_action(Q, state)
        score += reward
        pos_track.append(info["distance left"])
        vel_track.append(info["velocity"])
        acc_track.append(info["acceleration"])
        time_track.append(info["time"])
    env.close()

    x1 = time_track
    y1 = vel_track
    plt.figure(1)
    plt.plot(x1, y1)
    plt.savefig('vel-time.png')

    y2 = acc_track
    plt.figure(2)
    plt.plot(x1, y2)
    plt.savefig('acc-time.png')

    y3 = pos_track
    plt.figure(3)
    plt.plot(x1, y3)
    plt.savefig('pos-time.png')

    gc.collect()
