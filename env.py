import time
import numpy as np
#import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(2)


class Env( ):
    def __init__(self):
        super(Env, self).__init__()
        self.actions = [
            "Channel_1",
            "Channel_6",
            "Channel_11"
        ]
        self.n_actions = len(self.actions)
        #self.title('Channel Select')
        self.n_features = 4
        self.env_state = {
            "Channel_1":{},
            "Channel_2":{},
            "Channel_3":{}
        }
        self.state =""
        self.time = 1
        self.count = 1
        self.time_env_state = {
            1: {"Channel_1": np.array([-50, -60, 5, 30]),
                 "Channel_6": np.array([-20, -30, 25, 24]),
                 "Channel_11": np.array([-10, -20, 60, 24])},
            2: {"Channel_1": np.array([-50, -60, 10, 30]),
                 "Channel_6":  np.array([-30, -40, 20, 24]),
                 "Channel_11": np.array([-60, -70, 5, 24])
            },
            3: {"Channel_1":np.array([-40, -50, 15, 30]),
                 "Channel_6": np.array([-70, -80, 2, 24]),
                 "Channel_11": np.array([-15, -25, 60, 24])

            }
        }

    def reset(self):
        #self.update()
        #time.sleep(0.5)
        self.state = "Channel_1"
        # return observation
        return self.time_env_state[self.time][self.state]



    def step(self, action):
        self.time += 1
        print(self.time)
        next_state = action

        if next_state == self.state:
            self.count += 1
            reward = self.count
        else:
            reward = 0
            self.count = 1
        self.state = next_state
        return self.time_env_state[self.time][next_state], reward




