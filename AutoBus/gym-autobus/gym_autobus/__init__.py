from gym.envs.registration import register

register(
    id='autobus-v0',
    entry_point='gym_autobus.envs:AutobusEnv',
)
