import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from config import consts, args
import math

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()


    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon)), self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon)))
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)



class DQN(nn.Module):
    def __init__(self, n):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.n)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        # x = self.fc5(x)
        # for only positive rewards
        x = F.relu(self.fc5(x))
        return x


class DQNDueling(nn.Module):
    def __init__(self, n):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQNDueling, self).__init__()
        self.n = n

        self.conv1 = nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_h= nn.Linear(7 * 7 * 64, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512, self.n)

        self.init_weights()

    def init_weights(self):
        pass
        # self.fc_z_v.weight.data.zero_()
        # self.fc_z_a.weight.data.zero_()
        # self.fc_z_v.bias.data.zero_()
        # self.fc_z_a.bias.data.zero_()

    def forward(self, x):

        # image network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc_h(x.view(x.size(0), -1)))

        # value and advantage networks
        v, a = F.relu(self.fc_z_v(x)), self.fc_z_a(x)  # Calculate value and advantage streams
        a_mean = torch.stack(a.chunk(self.n, 1), 1).mean(1)
        x = v.repeat(1, self.n) + a - a_mean.repeat(1, self.n)  # Combine streams
        return x


class DVAN_ActionOut(nn.Module):
    def __init__(self, n):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DVAN_ActionOut, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_h= nn.Linear(7 * 7 * 64, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512, self.n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc_h(x.view(x.size(0), -1)))
        v, a = self.fc_z_v(x), self.fc_z_a(x)  # Calculate value and advantage streams
        return v, a

class DVAN_ActionIn(nn.Module):

    def __init__(self, n):

        super(DVAN_ActionIn, self).__init__()
        # self.mask = Variable(mask, requires_grad=False)
        self.conv1 = nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_h= nn.Linear(7 * 7 * 64, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512 + n, 1)

    def forward(self, s, a=None):

        # state CNN
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = F.relu(self.fc_h(s.view(s.size(0), -1)))

        # value network
        V = F.relu(self.fc_z_v(s))

        if a is None:
            return V

        if self.training:
            x = torch.cat([s, a.float()], dim=1)
            # advantage network
            A = self.fc_z_a(x)  # Calculate value and advantage streams

        else:
            s = s.unsqueeze(1)
            s = s.repeat(1, a.data.shape[1], 1)
            x = torch.cat([s, a], dim=2)
            A = self.fc_z_a(x)

        return V, A



class DQNNoise(nn.Module):
    def __init__(self, n):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQNNoise, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.n)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        # x = self.fc5(x)
        # for only positive rewards
        x = F.relu(self.fc5(x))
        return x


class BehavioralNet(nn.Module):

    def __init__(self, n):

        super(BehavioralNet, self).__init__()
        # self.mask = Variable(mask, requires_grad=False)
        # self.conv1 = nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2))
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1))
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1))
        # self.fc_h= nn.Linear(10 * 10 * 64, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_beta = nn.Linear(512, n)
        self.fc_z_q = nn.Linear(512+3, 1)
        self.fc_z_r = nn.Linear(512 + 3, 1)
        self.fc_z_p = nn.Linear(512 + 3, 512)

        # batch normalization and dropout
        self.conv1 = nn.Sequential(
            nn.Dropout2d(0.0),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.fc_h = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        )

    def forward(self, s, a=None):

        # state CNN
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = F.relu(self.fc_h(s.view(s.size(0), -1)))

        # value network
        v = F.leaky_relu(self.fc_z_v(s))

        # behavioral estimator
        beta = F.leaky_relu(self.fc_z_beta(s))

        if a is None:
            return v, beta, s

        phi = s.clone()

        if self.training:

            x = torch.cat([s, a], dim=1)

        else:
            s = s.unsqueeze(1)
            s = s.repeat(1, a.data.shape[1], 1)
            x = torch.cat([s, a], dim=2)

        q = F.leaky_relu(self.fc_z_q(x))
        r = F.leaky_relu(self.fc_z_r(x))
        p = F.leaky_relu(self.fc_z_p(x))

        return v, q, beta, r, p, phi


class BehavioralNetDeterministic(nn.Module):

    def __init__(self, n):

        super(BehavioralNetDeterministic, self).__init__()
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_beta = nn.Linear(512, 3)
        self.fc_z_r = nn.Linear(512 + 3, 1)
        self.fc_z_q = nn.Linear(512+3, 1)
        self.fc_z_p = nn.Linear(512 + 3, 512)

        # batch normalization and dropout
        self.conv1 = nn.Sequential(
            nn.Dropout2d(0.0),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.fc_h = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        )

    def fix_weights(self):
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.fc_h.weight.requires_grad = False
        self.fc_h.bias.requires_grad = False

    def forward(self, s, a=None):

        # state CNN
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = F.relu(self.fc_h(s.view(s.size(0), -1)))

        # value network
        v = F.relu(self.fc_z_v(s))

        # behavioral estimator
        beta = self.fc_z_beta(s)

        # 3-state tanh
        beta = 0.5 * F.tanh(beta*100 - 5) + 0.5 * F.tanh(beta*100 + 5)

        if a is None:
            a = beta.sign() * (beta.abs() > 0.5).float()

        x = torch.cat([s, a], dim=1)

        r = F.relu(self.fc_z_r(x))
        p = F.relu(self.fc_z_p(x))
        q = F.relu(self.fc_z_q(x))

        return v, q, beta, r, p, s


class BehavioralRNN(nn.Module):

    def __init__(self):

        super(BehavioralRNN, self).__init__()

        # history
        self.rnn = nn.GRU(512+3, 256, 1, batch_first=True, dropout=0, bidirectional=False)

        # linear estimators
        self.fc_z_v = nn.Linear(512 + 256, 1)
        self.fc_z_beta = nn.Linear(512 + 256, 3)
        self.fc_z_q = nn.Linear(512 + 256 + 3, 1)
        self.fc_z_r = nn.Linear(512 + 256 + 3, 1)
        self.fc_z_p = nn.Linear(512 + 256 + 3, 512)

        # batch normalization and dropout
        self.conv1 = nn.Sequential(
            nn.Dropout2d(0.0),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        )

        self.fc_h = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        )

    def forward(self, o, a, h, a_next=None):

        batch, seq, channels, height, width = o.shape
        o = o.view(batch*seq, channels, height, width)

        # state CNN
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        o = F.relu(self.fc_h(o.view(o.size(0), -1)))

        o = o.view(batch, seq, 512)

        # calc state history
        x = torch.cat([o, a], dim=2)
        s, next_h = self.rnn(x, h.unsqueeze(0))

        s_prev = torch.cat([h.usqueeze(1), s[:, :-1, :]], dim=1)

        x = torch.cat([o, s_prev], dim=2)
        # value network
        v = F.relu(self.fc_z_v(x))
        # behavioral estimator
        beta = self.fc_z_beta(x)

        if a_next is None:
            a_next = beta[:, -1, :].sign() * (beta[:, -1, :].abs() > 0.5).float()

        a_next = a_next.unsqueeze(1)
        a = torch.cat([a, a_next], dim=1)
        x = torch.cat([o, a, s_prev], dim=2)
        # reward prediction
        q = F.relu(self.fc_z_q(x))
        # reward prediction
        r = F.relu(self.fc_z_r(x))
        # state prediction
        p = F.relu(self.fc_z_p(x))

        return v, q, beta, p, r, next_h

class BehavioralHotNet(nn.Module):

    def __init__(self):

        super(BehavioralHotNet, self).__init__()

        # self.fc_z_f = nn.Linear(512 + 8, 64)
        # self.fc_z_q = nn.Linear(512 + 8 + 64, 1)
        # self.fc_z_r = nn.Linear(512 + 8 + 64, 1)
        # self.fc_z_p = nn.Linear(512 + 8 + 64, 512)

        # self.fc_z_f = nn.Bilinear(512, 8, 64)
        # self.fc_z_q = nn.Linear(64, 1)
        # self.fc_z_r = nn.Linear(64, 1)
        # self.fc_z_p = nn.Linear(64, 512)

        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_beta = nn.Linear(512, 18)

        self.fc_z_q = nn.Linear(512, 1)
        self.fc_z_r = nn.Linear(512, 1)
        self.fc_z_p = nn.Linear(512, 512)

        # batch normalization and dropout
        self.conv1 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.fc_h = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        )

        self.fc_a = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(8, 512),
            # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        )

        # initialization
        self.conv1[1].bias.data.zero_()
        self.conv2[1].bias.data.zero_()
        self.conv3[1].bias.data.zero_()

    def forward(self, s, a=None):

        # state CNN
        s = F.leaky_relu(self.conv1(s))
        s = F.leaky_relu(self.conv2(s))
        s = F.leaky_relu(self.conv3(s))
        s = F.leaky_relu(self.fc_h(s.view(s.size(0), -1)))

        # value network
        v = F.leaky_relu(self.fc_z_v(s))

        # behavioral estimator
        beta = F.leaky_relu(self.fc_z_beta(s))

        if a is None:
            return v, beta, s

        phi = s.clone()

        # if self.training:
        #
        #     x = torch.cat([s, a], dim=1)
        #     h = F.relu(self.fc_z_f(x))
        #     x = torch.cat([x, h], dim=1)
        #
        # else:
        #     s = s.unsqueeze(1)
        #     s = s.repeat(1, a.data.shape[1], 1)
        #     x = torch.cat([s, a], dim=2)
        #     h = F.relu(self.fc_z_f(x))
        #     x = torch.cat([x, h], dim=2)

        # if self.training:
        #
        #     x = F.relu(self.fc_z_f(s, a))
        #     q = F.leaky_relu(self.fc_z_q(x))
        #     r = F.leaky_relu(self.fc_z_r(x))
        #     p = F.leaky_relu(self.fc_z_p(x))
        #
        # else:
        #
        #     batch, seq, channels = a.shape
        #     s = s.repeat(seq, 1)
        #     a = a.view(batch*seq, channels)
        #
        #     x = F.relu(self.fc_z_f(s, a))
        #
        #     q = F.leaky_relu(self.fc_z_q(x))
        #     r = F.leaky_relu(self.fc_z_r(x))
        #     p = F.leaky_relu(self.fc_z_p(x))
        #
        #     q = q.view(batch, seq, 1)
        #     r = r.view(batch, seq, 1)
        #     p = p.view(batch, seq, 512)

        if not self.training:
            s = s.unsqueeze(1)

        a = F.leaky_relu(self.fc_a(a))
        x = s * a

        q = F.leaky_relu(self.fc_z_q(x))
        r = F.leaky_relu(self.fc_z_r(x))
        p = F.leaky_relu(self.fc_z_p(x))

        return v, q, beta, r, p, phi
