import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from config import consts, args


class DPiN(nn.Module):

    def __init__(self):

        super(DPiN, self).__init__()

        # policy estimator
        self.fc_pi = nn.Sequential(
            nn.Linear(512, 18),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            nn.Dropout(0.25),
            # nn.Dropout2d(0.25),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_h = nn.Sequential(
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # initialization
        self.cnn_conv1[1].bias.data.zero_()
        # self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def reset_target(self):
        self.fc_pi[0].weight.data.zero_()
        self.fc_pi[0].bias.data.zero_()

    def forward(self, s):

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        phi = s.detach()
        s = self.cnn_h(s.view(s.size(0), -1))

        # behavioral estimator
        pi = self.fc_pi(s)

        return pi, phi


class DQN(nn.Module):

    def __init__(self):

        super(DQN, self).__init__()

        # q-value net
        self.fc_q = nn.Sequential(
            nn.Linear(512, 18),
            # nn.ReLU()
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            nn.Dropout(0.25),
            # nn.Dropout2d(0.25),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_h = nn.Sequential(
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # initialization
        self.cnn_conv1[1].bias.data.zero_()
        # self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def reset_target(self):
        self.fc_q[0].weight.data.zero_()
        self.fc_q[0].bias.data.zero_()

    def forward(self, s):

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        phi = s.detach()
        s = self.cnn_h(s.view(s.size(0), -1))

        # behavioral estimator
        q = self.fc_q(s)

        return q, phi


class DVN(nn.Module):

    def __init__(self):

        super(DVN, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            nn.Linear(512, 1),
            # nn.ReLU()
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            nn.Dropout(0.25),
            # nn.Dropout2d(0.25),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_h = nn.Sequential(
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # initialization
        self.cnn_conv1[1].bias.data.zero_()
        # self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def reset_target(self):
        self.fc_v[0].weight.data.zero_()
        self.fc_v[0].bias.data.zero_()

    def forward(self, s):

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        phi = s.detach()
        s = self.cnn_h(s.view(s.size(0), -1))

        # behavioral estimator
        v = self.fc_v(s)

        return v, phi


class BetaNet(nn.Module):

    def __init__(self):

        super(BetaNet, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            # nn.ReLU()
        )

        # q-value net
        self.fc_q = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 18),
            # nn.ReLU()
        )

        # policy estimator
        self.fc_pi = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 18),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            nn.Dropout(0.25),
            # nn.Dropout2d(0.25),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.cnn_h = nn.Sequential(
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
        )

        # initialization
        self.cnn_conv1[1].bias.data.zero_()
        # self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def forward(self, s):

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        s = self.cnn_h(s.view(s.size(0), -1))
        phi = s.detach()

        # behavioral estimator
        v = self.fc_v(s)
        q = self.fc_q(s)
        pi = self.fc_pi(s)

        return pi, q, v, phi


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()

        # policy estimator
        self.fc_actor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 18),
        )

    def forward(self, phi):
        actor = self.fc_actor(phi)
        return actor


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        # policy estimator
        self.fc_critic = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 18),
        )

    def forward(self, phi):
        critic = self.fc_critic(phi)
        return critic
