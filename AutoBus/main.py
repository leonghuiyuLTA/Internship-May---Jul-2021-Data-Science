import autobus_env
from time import sleep

# set up the tiles (position, velocity, acceleration)






if __name__ == "__main__":
    env = autobus_env.AutobusEnv()
    init = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(1)
        print(info)
        sleep(0.1)

# for the rendering part NEED try to use the online version, so need to do full import, or just copy this file into the other
