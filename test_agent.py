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
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationSampler
from player import player_worker, QPlayer, AVPlayer, AVAPlayer

from agent import Agent


class TestAgent(Agent):

    def __init__(self):
        super(TestAgent, self).__init__()

        if not args.value_advantage:
            self.player = QPlayer
            models = {(0,): DQN, (1,): DQNDueling}
            Model = models[(self.dueling,)]
            self.model_single = Model(self.action_space)

        elif not args.input_actions:
            self.player = AVPlayer
            self.model_single = DVAN_ActionOut(self.action_space)
        else:
            self.player = AVAPlayer
            self.model_single = DVAN_ActionIn(3)

        # configure learning
        if args.cuda:
            self.model_single = self.model_single.cuda()
            self.model = torch.nn.DataParallel(self.model_single)
        else:
            self.model = self.model_single

    def resume(self, model_path):

        state = torch.load(model_path)
        self.model.load_state_dict(state['state_dict'])
        return state['aux']

    def play(self, n_tot):

        player = self.player()
        params = self.model.state_dict()

        for episode in range(n_tot):

            for step in player.test_play(params):

                yield step

