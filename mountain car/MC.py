import numpy as np
import gym
from matplotlib import pyplot as plt

# 20 intervals for each variable, overall will have 400 tiles
pos_intervals = np.linspace(-1.2, 0.6, 20)
vel_intervals = np.linspace(-0.07, 0.07, 20)
action_space = [0, 1, 2]


# obtain the state key (p,v) where p is the interval number in pos, v in vel
def get_state(res):
    pos, vel = res
    pos_key = np.digitize(pos, pos_intervals)
    vel_key = np.digitize(vel, vel_intervals)
    return pos_key, vel_key


def choose_action(epsilon, Q, state):
    # exploration
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    # exploitation
    max_index = 0
    max_val = -1e9
    for i in range(3):
        if Q[state, i] > max_val:
            max_val = Q[state, i]
            max_index = i
    return max_index


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    rounds = 10000
    alpha = 0.1
    gamma = 1
    epsilon = 1
    Q = {}
    states = []
    score_tracker = np.zeros(rounds)
    num_success = 0
    success_tracker = np.zeros(int(rounds/1000) + 1)
    # setup - initialise for all state action pairs
    for p in range(21):
        for v in range(21):
            states.append((p,v))
    for s in states:
        for a in action_space:
            Q[s, a] = 0

    # learning process
    for i in range(rounds):
        done = False
        score = 0
        # initialise S
        init_obs = env.reset()
        state = get_state(init_obs)
        # choose A from S using policy(epsilon greedy)
        action = choose_action(epsilon, Q, state)
        # repeat until S is terminal
        while not done:
            # Take action A, observe R, S'
            obs, reward, done, info = env.step(action)
            new_state = get_state(obs)
            # Choose A' from S' using policy(epsilon greedy)
            new_action = choose_action(epsilon, Q, new_state)
            # update Q
            Q[state, action] += alpha * (reward + gamma * Q[new_state, new_action] - Q[state, action])
            # update S, A
            state = new_state
            action = new_action
            score += reward
        score_tracker[i] = score
        # updating epsilon so agent does less random actions
        epsilon -= 1/rounds
        # printing stuff for user
        if score > -1000:
            num_success += 1
        if i % 100 == 0 and i > 0:
            print("Round: ", i, " Score: ", score_tracker[i], " Epsilon: %.3f" % epsilon)
        if i % 1000 == 0 and i > 0:
            print("Num of successes: ", num_success)
            success_tracker[int(i/1000)] = num_success - success_tracker[int(i/1000)-1]

    # mean for set of 50, for each round
    mean_score = np.zeros(rounds)
    for t in range(rounds):
        mean_score[t] = np.mean(score_tracker[max(0, t-50) : (t+1)])
    plt.plot(mean_score)
    plt.show()
