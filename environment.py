import gym
import numpy as np
from skimage import io, transform, color
import torch
import cv2

from config import consts, args
from preprocess import preprocess_screen

class Env(object):

    def __init__(self):

        self.env = gym.make(consts.gym_game_dict[args.game])
        self.i = 0  # Internal step counter
        self.T = args.max_episode_length
        self.lives = 0  # Life counter (used in DeepMind training)
        self.s, self.o, self.r, self.t, self.info = None, None, None, None, None
        self.buffer = [np.zeros((args.height, args.width), dtype=np.float32)] * args.skip
        self.meanings = self.env.env.get_action_meanings()

    def get_action(self, a):
        a = consts.activation2action[args.game][a]
        name = consts.action_meanings[a]
        return self.meanings.index(name)


    def to_buffer(self, o):
        self.buffer.pop()
        self.buffer.insert(0, o)

    def to_tensor(self):
        state = np.stack(self.buffer, axis=0)
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def reset(self):
        # Reset internals
        self.i = 0
        self.score = 0
        self.lives = self.env.env.ale.lives()
        # Process and return initial state
        self.env.reset()
        for j in range(args.history_length):
            self.step(consts.nop)

    def step(self, action):
        # Process state

        self.r = 0

        action = self.get_action(action)
        # print("Action chosen: %s" % self.meanings[action])
        # open ai-gym skips without asking

        self.o, self.r, self.t, self.info = self.env.step(action)
        self.score += self.r

        self.to_buffer(preprocess_screen(self.o))

        self.s = self.to_tensor()


        # # Detect loss of life as terminal in training mode
        # if self.training:
        #     lives = self.env.env.ale.lives()
        #     if lives < self.lives:
        #         done = True
        #     else:
        #         self.lives = lives
        #     # Time out episode if necessary
        #     self.t += 1
        if self.i == self.T:
            self.t = True

    def action_space(self):
        return self.env.action_space.n

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, close=False):
        if not close:
            self.env.render()
        else:
            self.env.render(close=True)

    def close(self):
        self.env.close()