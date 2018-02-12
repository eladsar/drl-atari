import gym
import numpy as np
import torch
import cv2

from config import consts, args
from preprocess import preprocess_screen


class RandomPlayer(object):

    def __init__(self, game):
        strategies = {'spaceinvagers': self.spaceinvaders,
                      'mspacman': self.mspacman,
                      'pinball': self.pinball,
                      'qbert': self.qbert,
                      'revenge': self.revenge,
                      }
        self.strategy = strategies[game]

    def spaceinvaders(self):

        pass

    def mspacman(self):
        pass

    def pinball(self):
        pass

    def qbert(self):
        pass

    def revenge(self):
        pass


class Env(object):

    def __init__(self, action_offset=None):

        self.env = gym.make(consts.gym_game_dict[args.game])
        self.i = 0  # Internal step counter
        self.T = args.max_episode_length
        self.lives = 0  # Life counter (used in DeepMind training)
        self.s, self.r, self.t, self.info = None, None, None, None
        self.buffer = [np.zeros((args.height, args.width), dtype=np.float32)] * args.history_length
        self.meanings = self.env.env.get_action_meanings()
        self.skip = args.skip

        if action_offset is None:
            self.action_offset = args.action_offset
        else:
            self.action_offset = action_offset

        self.action_meanings = consts.action_meanings
        # self.activation2action = consts.activation2action[args.game]
        self.mask = torch.LongTensor(consts.excitation_mask[args.game])
        self.reverse_excitation_map = consts.reverse_excitation_map
        self.actions_dict = consts.actions_dict[args.game]
        self.nop = consts.nop
        self.action_buffer = [self.nop] * (self.action_offset - 1)

    def get_action_from_buffer(self, a_in):
        self.action_buffer.insert(0, a_in)
        return self.action_buffer.pop()

    def get_action(self, a):
        name = self.actions_dict[self.action_meanings[a]]
        return self.meanings.index(name)

    # def get_action_output(self, a):
    #     a = self.activation2action[a]
    #     name = self.action_meanings[a]
    #     return self.meanings.index(name)

    def to_buffer(self, o):
        self.buffer.pop()
        self.buffer.insert(0, o)
        # self.buffer.pop(0)
        # self.buffer.append(o)

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
        o0 = self.env.reset()
        # enter the first and second frames to history
        o1, _, _, _ = self.env.step(self.nop)
        self.to_buffer(preprocess_screen([o0, o1]))

        for j in range(args.history_length - 1):
            self.step(self.nop)

    def simple_reset(self):
        # Reset internals
        self.score = 0
        # Process and return initial state
        self.env.reset()

    def simple_step(self, actions):

        self.r = 0

        for a in actions:
            a = self.get_action(a)
            x, r, self.t, self.info = self.env.step(a)
            self.r += r

        self.score += self.r

        return x, actions, self.r, self.t

    def step(self, action):

        # Process state
        action = self.get_action(action)
        # print("Action chosen: %s" % self.meanings[action])
        # open ai-gym skips without asking

        self.r = 0
        o = []
        # for i in range(self.skip * (self.action_offset - 1)):
        #     x, r, self.t, self.info = self.env.step(self.last_action)
        #     self.r += r
        #     o.append(x)

        for i in range(self.skip):
            x, r, self.t, self.info = self.env.step(self.get_action_from_buffer(action))
            self.r += r
            o.append(x)

        self.last_action = action
        self.score += self.r

        # i=0
        for i in range(0, len(o), self.skip):
            frame = preprocess_screen(o[i:i+2])
            self.to_buffer(frame)

        self.s = self.to_tensor()

        return x, action, self.r, self.t

        # # Detect loss of life as terminal in training mode
        # if self.training:
        #     lives = self.env.env.ale.lives()
        #     if lives < self.lives:
        #         done = True
        #     else:
        #         self.lives = lives
        #     # Time out episode if necessary
        #     self.t += 1
        # self.i += 1
        # if self.i == self.T:
        #     self.t = True

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
