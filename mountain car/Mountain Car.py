import numpy as np
from matplotlib import pyplot as plt
import gym

pos_bins = np.linspace(-1.2, 0.6, 20)
vel_bins = np.linspace(-0.07, 0.07, 20)

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
    env._max_episode_steps = 1000
    rounds = 50000
    learn_rate = 0.1
    discount = 1
    epsilon = 1
    Q = {}
    states = []
    for p in range(21):
        for v in range(21):
            states.append((p,v))
    for state in states:
        for action in [0,1,2]:
            Q[state,action] = 0
    numsuccess = 0
    scoretracker = np.zeros(rounds)
    for i in range(rounds):
        done = False
        init = env.reset()
        state = get_state(init)
        score = 0
        while not done:
            action = chooseaction(epsilon, Q, state)
            obs, reward, done, info = env.step(action)
            #need to include the Q-value update part here
            newstate = get_state(obs)
            newaction = chooseaction(0,Q,newstate)
            Q[state, action] += learn_rate*(reward + discount*Q[newstate, newaction] - Q[state,action])
            score += reward
            state = newstate
        scoretracker[i] = score
        epsilon -= 2 / rounds
        if score > -1000: numsuccess += 1
        if i % 100 == 0: print("Round: " + str(i) + " Score: " + str(scoretracker[i]) + " Epsilon : %.3f" % epsilon)
        if i % 1000 == 0: print("no of success: " + str(numsuccess))
