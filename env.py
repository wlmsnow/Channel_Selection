import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['s', 'd', 'l']
        self.n_actions = len(self.action_space)
        self.title('Channel Select')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []
        self.actions = {
            "Channel_1":{},
            "Channel_6":{},
            "Channel_11":{}
        }

       # tk.mainloop()

    def _build_canvas(self):
        """
        Create the Gridworld
        #(rectangle represents channel 1)
        #(triangle represents channel 6 )
        #(circle represents channel 11)
        """
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # add img to canvas
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])#(rectangle represents channel 1)
        self.triangle = canvas.create_image(150, 50, image=self.shapes[1])#(triangle represents channel 6 )
        self.circle = canvas.create_image(50, 150, image=self.shapes[2])#(circle represents channel 11)
        self.ovel = canvas.create_oval(30,30,70,70,fill='pink' )
        # pack all
        canvas.pack()

        return canvas
    def E_State(self):
        self.E_state = {"Channel_1":[],
                   "Channel_6":[],
                   "Channel_11":[]
                   }

    def update_E_State(self,list1,list2,list3):
        self.E_state["Channel_1"] = list1
        self.E_state["Channel_6"] = list2
        self.E_state["Channel_11"] = list3

    def state_action(self,action):

        self.actions["Channel_1"] = {"right": (1, 0),"down": (0, -1)}
        self.actions["Channel_6"] = {"left": (-1, 0),"down_left":(-1,-1)}
        self.actions["Channel_11"]= { "up": (0, 1),"up_right":(1,1)}

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def coords_to_state(self, coords):
        """
        Coordinates to state
        """
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def state_to_coords(self, state):
        """
         state to Coordinates 
      
        """
        x = int(state[0] * 100 + 50)
        y = int(state[1] * 100 + 50)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.ovel)
        self.canvas.move(self.ovel, UNIT / 2 - x, UNIT / 2 - y)
        self.render()
        # return observation
        return self.coords_to_state(self.canvas.coords(self.ovel))


    def step(self, action):
        state = self.canvas.coords(self.ovel)
        delta = np.array([0, 0])
        self.render()

        if action == 0:  # up
            delta = (0, -100)
        elif action == 1:  # down
            delta = (0, 100)
        elif action == 2:  # left
            delta = (-100, 0)
        elif action == 3:  # right
            delta = (100, 0)
        elif action == 4:  # left_down
            delta = (-100, 100)
        elif action == 5:  # right_up
            delta = (100, -100)
        elif action == 6:  # constant
            delta = (0, 0)

        # move agent
        self.canvas.move(self.ovel, delta[0], delta[1])
        # move rectangle to top level of canvas
        #self.canvas.tag_raise(self.ovel)
        next_state = self.canvas.coords(self.ovel)

        # reward function
        if next_state == self.canvas.coords(self.circle):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.triangle),
                            self.canvas.coords(self.triangle)]:
            reward = -100
            done = True
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()
if  __name__ == "__main__":
    env = Env()
    """agent.state = {"Channel_1":[],
                   "Channel_6":[],
                   "Channel_11":[]}
"""
    #for episode in range(4):
       #env.reset()
    while True:
        time.sleep(1)
        env.step(3) #right
        time.sleep(1)
        env.step(4) #left_down
        time.sleep(1)
        env.step(0) #up
        time.sleep(1)
        env.step(6)#constant
        time.sleep(1)
        env.step(1)#down
        time.sleep(1)
        env.step(5)#right_up
        time.sleep(1)
        env.step(2)#left


