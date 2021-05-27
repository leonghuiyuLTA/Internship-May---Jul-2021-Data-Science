import gym
import numpy as np
import math
from matplotlib import pyplot as plt
from time import sleep
# states needs to be created
action_space = [0, 1]

# state areas
# position: -4.8 to 4.8, split into 3 zones < -1.6, -1.6 < x < 1.6, > 1.6
# velocity: -inf to inf, split into 3 zones < -1, -1 < x < 1, > 1
# angle: -24 to 24(need to convert received data), split into 6 zones, -12 to -6 to -1 to 0 to 1 to 6 to 12
# angle velocity: -inf to inf, split into 3 zones < -10, -10 < x < 10, > 10


def get_state(obs):
    pos, vel, ang, avl = obs
    ang = 180/math.pi * ang
    # discretize position
    if pos < -0.8:
        p_state = 0
    elif pos > 0.8:
        p_state = 2
    else: p_state = 1
    # discretize velocity
    if vel < -1:
        v_state = 0
    elif vel > 1:
        v_state = 2
    else:
        v_state = 1
    # discretize angle
    if ang < -6:
        a_state = 0
    elif ang < -1:
        a_state = 1
    elif ang < 0:
        a_state = 2
    elif ang < 1:
        a_state = 3
    elif ang < 6:
        a_state = 4
    else:
        a_state = 5
    # discretize anglular velocity
    if avl < -50:
        av_state = 0
    elif avl > 50:
        av_state = 2
    else:
        av_state = 1
    return p_state, v_state, a_state, av_state


def choose_action(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    if Q[state, 0] > Q[state, 1]:
        return 0
    return 1


def update_values(SA, Q, amt):
    for st in reversed(SA):
        Q[st] += 0.1 * (0.8 * amt - Q[st])
        amt = Q[st]


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    rounds = 100
    alpha = 0.1
    gamma = 0.8
    epsilon = 0.8
    Q = {}
    states = []
    score_tracker = np.zeros(rounds)

    # setting up the state space
    for p in range(3):
        for v in range(3):
            for a in range(6):
                for av in range(3):
                    states.append((p, v, a, av))

    # set up value referencing
    for s in states:
        for a in action_space:
            Q[s, a] = 0

    for i in range(rounds):
        print("Round: " , i)
        done = False
        score = 0
        SA = []
        obs = env.reset()
        state = get_state(obs)
        action = choose_action(state, Q, epsilon)
        while not done:
            observation, reward, done, info = env.step(action)
            new_state = get_state(observation)
            new_action = choose_action(new_state, Q, epsilon)
            state = new_state
            action = new_action
            score += reward
            SA.append((state, action))
        update_values(SA, Q, score)
        score_tracker[i] = score

    x = np.arange(0, rounds)
    plt.plot(x, score_tracker)
    plt.show()
