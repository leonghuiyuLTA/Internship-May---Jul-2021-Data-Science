import gc
import autobus_env
import numpy as np
import pickle
from matplotlib import pyplot as plt
import time


# set up the tiles (position, velocity, acceleration)
pos_intervals = np.linspace(0, 260, 260)
vel_intervals = np.linspace(0, 60, 60)
acc_intervals = np.linspace(-4, 5, 10)
action_space = np.arange(-4, 5, 1)


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
    max_acc = -4
    max_val = -1e9
    for i in action_space:
        if Q[state, i] > max_val:
            max_val = Q[state, i]
            max_acc = i
    return max_acc


if __name__ == "__main__":
    env = autobus_env.AutobusEnv()
    rounds = 40000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.15
    Q = {} # 14million key-value pairs
    states = []
    scores = np.zeros(rounds)
    prev_score = 0.0
    tol = 0.005
    tol_count = 5
    conv_count = 0

    for p in range(262):
        for v in range(61):
            for a in range(11):
                states.append((p, v, a))

    for s in states:
        for a in action_space:
            Q[s, a] = 0
    #
    # with open("policy", 'rb') as fo:
    #     Q = pickle.load(fo)
    #     fo.close()

    print("starting")
    start = time.time()
    for i in range(rounds):
        if (i + 1) % 500 == 0:
            print("Round: ", i + 1)
        done = False
        score = 0
        init = env.reset()
        state = get_state(init)
        action = choose_action(epsilon, Q, state)
        while not done:
            obs, reward, done, info = env.step(action)
            new_state = get_state(obs)
            new_action = choose_action(epsilon, Q, new_state)
            Q[state, action] += alpha * (reward + gamma * Q[new_state, new_action] - Q[state, action])
            # update S, A
            state = new_state
            action = new_action
            score += reward
        if (i + 1) % 500 == 0:
            print(score)
        scores[i] = score
        if(abs(score - prev_score) < tol): 
            conv_count +=1 
            if(conv_count == tol_count):
                break 
        prev_score = score
        #epsilon -= 1 / (rounds-2)
    end = time.time()
    print("Time elapsed: ", end - start)

    filename = "policy"
    with open(filename, 'wb') as fo:
        pickle.dump(Q, fo)
        fo.close()

    mean_score = np.zeros(rounds)
    for t in range(rounds):
        mean_score[t] = np.mean(scores[max(0, t - 50): (t + 1)])
    plt.plot(mean_score)
    plt.savefig('mean_score.png')

    gc.collect()





