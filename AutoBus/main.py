import autobus_env
import numpy as np
import pickle
from matplotlib import pyplot as plt
import time
import pandas as pd

# set up the tiles (position, velocity, acceleration)
pos_intervals = np.linspace(0, 201, 201)
vel_intervals = np.linspace(0, 60, 120)
acc_intervals = np.linspace(-4, 4, 16)
action_space = np.arange(-4, 4, 0.5)


def get_state(obs):
    pos, vel, acc = obs
    pos_key = np.digitize(pos, pos_intervals)
    vel_key = np.digitize(vel, vel_intervals)
    acc_key = np.digitize(acc, acc_intervals)
    return pos_key, vel_key, acc_key


def choose_action(epsilon, Q, state):
    # exploration
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    # exploitation
    max_index = 0
    max_val = -1e9
    for i in action_space:
        if Q[state, i] > max_val:
            max_val = Q[state, i]
            max_index = i
    return max_index

# NEED TO CHANGE THE dt IN THE AUTOBUS ENV BECAUSE EVERY 0.1S UPDATE IS TOO FAST, DOESNT ALLOW ENOUGH LEEWAY TO ENTER NEW STATES
# ALSO, THE TILING HAS ISSUES BECAUSE CANNOT SAVE INTO A CSV FILE SO NEED TO RETHINK HOW TO CHOOSE STATES
# 1st trial: 6230s, but policy cannot be saved because too big of a file
if __name__ == "__main__":
    env = autobus_env.AutobusEnv()
    rounds = 1
    alpha = 0.1
    gamma = 0.8
    epsilon = 1
    Q = {} # 14million key-value pairs
    states = []
    scores = np.zeros(rounds)

    for p in range(201):
        for v in range(121):
            for a in range(17):
                states.append((p,v,a))

    for s in states:
        for a in action_space:
            Q[s, a] = 0

    print("starting")
    start = time.time()
    for i in range(rounds):
        print("Round: ", i + 1)
        done = False
        score = 0
        init = env.reset()
        state = get_state(init)
        action = choose_action(epsilon, Q, state)
        while not done:
            env.render()
            obs, reward, done, info = env.step(action)
            new_state = get_state(obs)
            new_action = choose_action(epsilon, Q, new_state)
            Q[state, action] += alpha * (reward + gamma * Q[new_state, new_action] - Q[state, action])
            # update S, A
            state = new_state
            action = new_action
            score += reward
            epsilon -= 0.05
        env.close()
        print(score)
        end = time.time()
        print("Time elapsed: ", end-start)

    dfp1 = pd.DataFrame(Q, index = [0]).T
    dfp1.to_csv("buspolicy.csv")



