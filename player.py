import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from config import consts, args
from model import DQN, DQNDueling, DVAN_ActionIn, DVAN_ActionOut
from environment import Env


class Player(object):
    def __init__(self):
        self.env = Env()
        self.greedy = args.greedy

    def play(self, params):
        raise NotImplementedError

    def close(self):
        self.env.close()


class QPlayer(Player):
    def __init__(self):
        super(QPlayer, self).__init__()
        self.model = DQN(consts.n_actions[args.game])
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)
        self.action_space = consts.n_actions[args.game]

    def play(self, params):

        self.model.load_state_dict(params)
        self.env.reset()
        softmax = torch.nn.Softmax()
        choices = np.arange(self.action_space, dtype=np.int)

        while not self.env.t:

            s = Variable(self.env.s, requires_grad=False)

            a = self.model(s)
            if self.greedy:
                a = a.data.cpu().numpy()
                a = np.argmax(a)
            else:
                a = softmax(a).data.squeeze(0).cpu().numpy()
                # print(a)
                a = np.random.choice(choices, p=a)
            self.env.step(a)

        return self.env.score


class AVPlayer(Player):
    def __init__(self):
        super(AVPlayer, self).__init__()
        self.model = DVAN_ActionOut(consts.n_actions[args.game])
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)
        self.action_space = consts.n_actions[args.game]

    def play(self, params):

        self.model.load_state_dict(params)
        self.env.reset()
        softmax = torch.nn.Softmax()
        choices = np.arange(self.action_space, dtype=np.int)

        while not self.env.t:

            s = Variable(self.env.s, requires_grad=False)

            v, a = self.model(s)
            if self.greedy:
                a = a.data.cpu().numpy()
                a = np.argmax(a)
            else:
                a = softmax(a).data.squeeze(0).cpu().numpy()
                a = np.random.choice(choices, p=a)
            self.env.step(a)

        return self.env.score


class AVAPlayer(Player):
    def __init__(self):
        super(AVAPlayer, self).__init__()
        self.model = DVAN_ActionIn(3)
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)
        self.action_space = consts.action_space

        excitation = torch.LongTensor(consts.excitation_map)
        mask = torch.LongTensor(consts.excitation_mask[args.game])
        mask_dup = mask.unsqueeze(0).repeat(consts.action_space, 1)
        actions = Variable(mask_dup * excitation, requires_grad=False)
        actions = actions.cuda()

        self.actions_matrix = actions.unsqueeze(0)
        self.actions_matrix = self.actions_matrix.repeat(1, 1, 1).float()

    def play(self, params):

        self.model.load_state_dict(params)
        self.model.eval()
        self.env.reset()
        softmax = torch.nn.Softmax()
        choices = np.arange(self.action_space, dtype=np.int)

        while not self.env.t:

            s = Variable(self.env.s, requires_grad=False)
            v, a = self.model(s, self.actions_matrix)
            a = a.squeeze(2)

            if self.greedy:
                a = a.data.cpu().numpy()
                a = np.argmax(a)
            else:
                a = softmax(a).data.squeeze(0).cpu().numpy()
                a = np.random.choice(choices, p=a)
            self.env.step(a)

        return self.env.score

    def test_play(self, params):

        self.model.load_state_dict(params)
        self.model.eval()
        self.env.reset()
        softmax = torch.nn.Softmax()
        choices = np.arange(self.action_space, dtype=np.int)

        # self.env.render()
        while not self.env.t:

            s = Variable(self.env.s, requires_grad=False)
            v, a = self.model(s, self.actions_matrix)
            a = a.squeeze(2)

            if self.greedy:
                a = a.data.cpu().numpy()
                a = np.argmax(a)
            else:
                a = softmax(a).data.squeeze(0).cpu().numpy()
                a = np.random.choice(choices, p=a)

            yield self.env.step(a)

        # self.env.render(close=True)



def player_worker(queue, jobs, done, Player):

    player = Player()

    while not done.is_set():
        params = jobs.get()
        score = player.play(params)
        queue.put(score)
    player.close()