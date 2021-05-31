import os
import numpy as np
from gym import spaces
from gym.utils import seeding
# VERSION 1 : BUS SUPPOSED TO DRIVE AT MAX SPEED UNTIL REACH GOAL

class AutobusEnv():
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.speed_limit = 50
        self.track_length = 3000
        self.dt = 0.1

        self.reward_weights = [1.0, 1.0] # 1 for reaching curr speed limit, 1.0 for jerk?
        self.seed()

        self.position = 0.0
        self.velocity = 0.0
        self.prev_acceleration = 0.0
        self.jerk = 0.0 # new acc - old acc
        self.time = 0
        self.done = False

        self.action_space = spaces.Box(low = -4, high = 4, shape = (1,))

        self.viewer = None

    def step(self,action):
        # assert self.action_space.contains(action), f'{action} ({type(action)}) invalid shape or bounds'
        self.position += (0.5 * action * self.dt ** 2 + self.velocity * self.dt)
        self.velocity += action * self.dt
        self.time += self.dt
        self.jerk = abs((action - self.prev_acceleration))
        self.prev_acceleration = action

        self.done = bool(self.position >= self.track_length)
        reward_list = self.get_reward() # is a list, but now we only care speed limits
        info = {
            'position' : self.position,
            'velocity' : self.velocity,
            'acceleration': self.prev_acceleration,
            'jerk': self.jerk,
            'time': self.time,
        }
        state = self.get_state()
        reward = np.array(reward_list).dot(np.array(self.reward_weights))
        return state, reward, self.done, info

    def get_state(self):
        return np.hstack((self.velocity, self.prev_acceleration))

    def get_reward(self):
        reward_forward = abs(self.velocity - self.speed_limit)
        reward_forward /= self.speed_limit

        reward_jerk = self.jerk / 5      # 5 is random number
        reward_list = [
            -reward_forward, -reward_jerk
        ]
        return reward_list

    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        self.prev_acceleration = 0.0
        self.jerk = 0.0  # new acc - old acc
        self.time = 0
        self.done = False
        state = self.get_state()
        return state

    def render(self, mode = 'human'):
        screen_width = 1000
        screen_height = 450
        clearance_x = 80
        clearance_y = 10
        zero_x = 0.25 * screen_width
        visible_track_length = 1000
        scale_x = screen_width / visible_track_length

        if self.viewer is None:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import rendering
            self.viewer = rendering.Viewer(width=screen_width,
                                           height=screen_height)

            rel_dir = os.path.join(os.path.dirname(__file__),
                                   'img')
            # start and finish line
            fname = os.path.join(rel_dir, 'start_finish_30x100.png')
            start = rendering.Image(fname,
                                    rel_anchor_y=0,
                                    batch=self.viewer.batch,
                                    group=self.viewer.background)
            start.position = (zero_x, clearance_y)
            self.viewer.components['start'] = start
            finish = rendering.Image(fname,
                                     rel_anchor_y=0,
                                     batch=self.viewer.batch,
                                     group=self.viewer.background)
            finish.position = (zero_x + scale_x * self.track_length,
                               clearance_y)
            self.viewer.components['finish'] = finish

            self.viewer.components['signs'] = []

            # speedometer
            fname = os.path.join(rel_dir, 'speedometer_232x190.png')
            speedometer = rendering.Image(fname,
                                          rel_anchor_y=0,
                                          batch=self.viewer.batch,
                                          group=self.viewer.background)
            speedometer.position = (screen_width - 110 - clearance_x,
                                    220 + clearance_y)
            self.viewer.components['speedometer'] = speedometer

            fname = os.path.join(rel_dir, 'needle_6x60.png')
            needle = rendering.Image(fname,
                                     rel_anchor_y=0.99,
                                     batch=self.viewer.batch,
                                     group=self.viewer.foreground)
            needle.position = (screen_width - 110 - clearance_x,
                               308 + clearance_y)
            self.viewer.components['needle'] = needle

            fname = os.path.join(rel_dir, 'needle_6x30.png')
            needle_sl = rendering.Image(fname,
                                        rel_anchor_y=2.6,
                                        batch=self.viewer.batch,
                                        group=self.viewer.background)
            needle_sl.position = (screen_width - 110 - clearance_x,
                                  308 + clearance_y)
            self.viewer.components['needle_sl'] = needle_sl

            # info figures
            self.viewer.history['velocity'] = []
            self.viewer.history['speed_limit'] = []
            self.viewer.history['position'] = []
            self.viewer.history['acceleration'] = []
            sns.set_style('whitegrid')
            self.fig = plt.Figure((640 / 80, 200 / 80), dpi=80)
            info = rendering.Figure(self.fig,
                                    rel_anchor_x=0,
                                    rel_anchor_y=0,
                                    batch=self.viewer.batch,
                                    group=self.viewer.background)
            info.position = (clearance_x - 40, 225 + clearance_y)
            self.viewer.components['info'] = info

            # car
            fname = os.path.join(rel_dir, 'bus.png')
            car = rendering.Image(fname,
                                  rel_anchor_x=1,
                                  batch=self.viewer.batch,
                                  group=self.viewer.foreground)
            car.position = (zero_x, 50 + clearance_y)
            self.viewer.components['car'] = car

            # speedometer
            deg = 60.0 + self.speed_limit * 3.6 * 1.5  # 1km/h =^ 1.5°
            self.viewer.components['needle_sl'].rotation = deg
            deg = 60.0 + self.velocity * 3.6 * 1.5  # 1km/h =^ 1.5°
            self.viewer.components['needle'].rotation = deg
            # self.viewer.components['velocity_label'].text = str(
            #     int(self.velocity * 3.6))
            # m, s = divmod(self.time, 60)
            # self.viewer.components['time_label'].text = f'{m:02.0f}:{s:02.0f}'

            # info figures
            self.viewer.history['velocity'].append(self.velocity * 3.6)
            self.viewer.history['speed_limit'].append(self.speed_limit *
                                                      3.6)
            self.viewer.history['position'].append(self.position)
            self.viewer.history['acceleration'].append(self.prev_acceleration)
            self.viewer.components['info'].visible = False
            if self.viewer.plot_fig or mode == 'rgb_array':
                self.viewer.components['info'].visible = True
                self.fig.clf()
                ax = self.fig.add_subplot(121)
                ax.plot(self.viewer.history['position'],
                        self.viewer.history['velocity'],
                        lw=2,
                        color='k')
                ax.plot(self.viewer.history['position'],
                        self.viewer.history['speed_limit'],
                        lw=1.5,
                        ls='--',
                        color='r')
                ax.set_xlabel('Position in m')
                ax.set_ylabel('Velocity in km/h')
                ax.set_xlim(
                    (0.0, max(500, self.position + (500 - self.position) % 500)))
                ax.set_ylim((0.0, 130))

                ax2 = self.fig.add_subplot(122)
                ax2.plot(self.viewer.history['position'],
                         self.viewer.history['acceleration'],
                         lw=2,
                         color='k')
                ax2.set_xlabel('Position in m')
                ax2.set_ylabel('Acceleration in m/s²')
                ax2.set_xlim(
                    (0.0, max(500, self.position + (500 - self.position) % 500)))
                ax2.set_ylim((-5.0, 5.0))

                self.fig.tight_layout()
                self.viewer.components['info'].figure = self.fig

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def reset_viewer(self):
        if self.viewer is not None:
            for key in self.viewer.history:
                self.viewer.history[key] = []
            self.viewer.components['signs'] = []

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]