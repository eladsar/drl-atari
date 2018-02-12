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


class BehavioralHotNet(nn.Module):

    def __init__(self):

        super(BehavioralHotNet, self).__init__()

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

        if not self.training:
            s = s.unsqueeze(1)

        a = F.leaky_relu(self.fc_a(a))
        x = s * a

        q = F.leaky_relu(self.fc_z_q(x))
        r = F.leaky_relu(self.fc_z_r(x))
        p = F.leaky_relu(self.fc_z_p(x))

        return v, q, beta, r, p, phi


class BehavioralDistNet(nn.Module):

    def __init__(self):

        super(BehavioralDistNet, self).__init__()

        # policy estimators
        self.on_beta = nn.Linear(512, 18)
        # self.on_pi_l = nn.Linear(512, 18)
        # self.on_pi_s = nn.Linear(512, 18)
        # self.on_pi_tau_s = nn.Linear(512, 18)
        # self.on_pi_tau_l = nn.Linear(512, 18)

        # value estimator
        self.on_vs = nn.Linear(512, args.atoms_short)
        self.on_vl = nn.Linear(512, args.atoms_long)

        self.on_qs = nn.Linear(512, args.atoms_short)
        self.on_ql = nn.Linear(512, args.atoms_long)

        self.on_pi_l = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 18),
        )

        self.on_pi_s = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 18),
        )

        self.on_pi_tau_s = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 18),
        )

        self.on_pi_tau_l = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 18),
        )

        # batch normalization and dropout
        self.rn_conv1 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
        )

        self.rn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.rn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )

        self.rn_h = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        )

        self.rn_a = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(8, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        )

        # initialization
        self.rn_conv1[1].bias.data.zero_()
        self.rn_conv2[1].bias.data.zero_()
        self.rn_conv3[1].bias.data.zero_()

        self.wasserstein = args.wasserstein

    def forward(self, s, a=None):

        # state CNN
        s = F.relu(self.rn_conv1(s))
        s = F.relu(self.rn_conv2(s))
        s = F.relu(self.rn_conv3(s))
        s = F.relu(self.rn_h(s.view(s.size(0), -1)))

        # value network
        vs = self.on_vs(s)
        vl = self.on_vl(s)

        # behavioral estimator
        beta = self.on_beta(s)
        pi_s_tau = self.on_pi_tau_s(s)
        pi_l_tau = self.on_pi_tau_l(s)
        pi_s = self.on_pi_s(s)
        pi_l = self.on_pi_l(s)

        phi = s.clone()

        if not self.training:
            s = s.unsqueeze(1)
            dim = 2

            batch, seq, channels = a.shape
            a = a.view(batch * seq, channels)
            a = F.leaky_relu(self.rn_a(a))
            a = a.view(batch, seq, 512)

        else:
            dim = 1
            a = F.leaky_relu(self.rn_a(a))

        x = s * a

        # q network
        qs = self.on_qs(x)
        ql = self.on_ql(x)

        if self.wasserstein:
            vs = F.softmax(vs, 1)
            vl = F.softmax(vl, 1)
            qs = F.softmax(qs, dim)
            ql = F.softmax(ql, dim)

        return vs, vl, beta, qs, ql, phi, pi_s, pi_l, pi_s_tau, pi_l_tau


class CriticQN(nn.Module):

    def __init__(self):

        super(CriticQN, self).__init__()

        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(512, args.atoms_short),
            nn.ReLU(),
            nn.Linear(256, args.atoms_short),
            nn.ReLU(),
        )

    def forward(self, s):
        return self.net(s)


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()

        self.critic_adv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1024, args.atoms_short),
            nn.LogSoftmax(1),
        )

        self.critic_v = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, args.atoms_long),
            nn.LogSoftmax(1),
        )

        # policy estimator
        self.fc_actor_f = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

        self.fc_actor_v = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )

        self.fc_actor_h = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )

    def init_target(self):
        self.critic_adv[1].bias.data.zero_()
        self.critic_adv[1].weight.data.zero_()
        self.critic_v[1].bias.data.zero_()
        self.critic_v[1].weight.data.zero_()

        self.critic_adv[1].bias.data[0] = 1
        self.critic_v[1].bias.data[0] = 1

    def forward(self, x):

        s = x[:, :, :512]
        critic_v = self.critic_v(s)

        # behavioral estimator
        pi_f = self.fc_actor_f(s)
        pi_v = self.fc_actor_v(s)
        pi_h = self.fc_actor_h(s)
        actor_pi = torch.cat([pi_f, pi_v, pi_h], dim=1)

        batch, seq, channels = x.shape

        adv = self.fc_adv(x.view(batch * seq, channels))
        critic_q = F.conv1d(critic_v, adv, padding=2*args.atoms_short)
        critic_q = critic_q.view(batch, seq, 2 * args.atoms_short + args.atoms_long)

        return actor_pi, critic_v, critic_q


class BehavioralDistEmbedding(nn.Module):

    def __init__(self):

        super(BehavioralDistEmbedding, self).__init__()

        # policy estimator
        self.fc_beta_f = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

        self.fc_beta_v = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )

        self.fc_beta_h = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )

        # value net
        self.fc_v = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, args.atoms_long),
            nn.LogSoftmax(1),
        )

        # advantage net
        # notice that since we do not flip the value in the convolution operation between the two distributions,
        # adv is expected to learn a reverse representation of the discretized support
        self.fc_adv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1024, 2*args.atoms_short + 1),
            nn.LogSoftmax(1),
        )

        # embedding

        self.emb_i_f = nn.Embedding(2, 172)
        self.emb_i_v = nn.Embedding(2, 170),
        self.emb_i_h = nn.Embedding(2, 170)

        self.emb_o = nn.Sequential(
            nn.Dropout(0.1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            nn.Dropout2d(0.2),
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
            nn.Dropout(0.1),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        # initialization
        self.cnn_conv1[1].bias.data.zero_()
        self.cnn_conv2[1].bias.data.zero_()
        self.cnn_conv3[1].bias.data.zero_()

    def forward(self, s, a):

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        s = self.cnn_h(s.view(s.size(0), -1))

        # value network
        v = self.fc_v(s)

        # behavioral estimator
        beta_f = self.fc_beta_f(s)
        beta_v = self.fc_beta_v(s)
        beta_h = self.fc_beta_h(s)
        beta = torch.cat([beta_f, beta_v, beta_h], dim=1)

        # embedding net
        a_f = a[:, :, 0].unsqueeze(1)
        a_v = a[:, :, 1].unsqueeze(1)
        a_h = a[:, :, 2].unsqueeze(1)

        a = torch.cat([self.emb_i_f(a_f), self.emb_i_v(a_v), self.emb_i_h(a_h)], dim=2)
        batch, seq, channels = a.shape
        a = self.emb_o(a.view(batch * seq, channels))
        a = a.view(batch, seq, 512)

        # aggregation
        s = s.unsqueeze(1)
        s = s.repeat(1, a.shape[1], 1)
        x = torch.cat([s, a], dim=2)
        batch, seq, channels = x.shape

        # advantage net
        adv = self.fc_adv(x.view(batch * seq, channels))

        # convolving with v
        q = F.conv1d(v, adv, padding=2*args.atoms_short)
        q = q.view(batch, seq, 2 * args.atoms_short + args.atoms_long)

        return beta, v, q, x


class ACDQN(nn.Module):

    def __init__(self):

        super(ACDQN, self).__init__()

        # prediction

        # self.fc_p = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.LeakyReLU()
        # )

        # policy estimator
        self.fc_pi = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(512, 18),
        )

        # value net
        self.fc_v = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.ReLU()
        )

        # advantage net
        # notice that since we do not flip the value in the convolution operation between the two distributions,
        # adv is expected to learn a reverse representation of the discretized support
        self.fc_adv = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(1024, 1),
        )

        # embedding

        self.emb_i_f = nn.Embedding(2, 172)
        self.emb_i_v = nn.Embedding(3, 170)
        self.emb_i_h = nn.Embedding(3, 170)

        self.emb_o = nn.Sequential(
            # nn.Dropout(0.1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            # nn.Dropout2d(0.2),
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
            # nn.Dropout(0.1),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def reset_target(self):
        self.fc_pi[0].weight.data.zero_()
        self.fc_pi[0].bias.data.zero_()
        self.fc_v[0].weight.data.zero_()
        self.fc_v[0].bias.data.zero_()
        self.fc_adv[0].weight.data.zero_()
        self.fc_adv[0].bias.data.zero_()

    def forward(self, s, a, beta=None):

        A = a.shape[1]

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        s = self.cnn_h(s.view(s.size(0), -1))
        phi = s

        # p = self.fc_p(s)

        # value network
        v = self.fc_v(s)

        # behavioral estimator
        pi = self.fc_pi(s)

        # embedding net
        a_f = a[:, :, 0].long()
        a_v = a[:, :, 1].long()
        a_h = a[:, :, 2].long()

        a = torch.cat([self.emb_i_f(a_f), self.emb_i_v(a_v), self.emb_i_h(a_h)], dim=2)
        batch, seq, channels = a.shape
        a = self.emb_o(a.view(batch * seq, channels))
        a = a.view(batch, seq, 512)

        # aggregation
        s = s.unsqueeze(1)
        s = s.repeat(1, A, 1)
        x = torch.cat([s, a], dim=2)
        batch, seq, channels = x.shape

        # advantage net
        adv = self.fc_adv(x.view(batch * seq, channels))
        adv = adv.view(batch, seq, 1)

        # if beta is not None:
        #     beta = F.softmax(beta.unsqueeze(2), dim=1)
        # else:
        #     beta = F.softmax(pi.detach().unsqueeze(2), dim=1)
        #     # q = v - 1. / A * adv.sum(1)

        beta = 1. / A * Variable(torch.ones(32, A, 1), requires_grad=False).cuda()
        q = v - (beta * adv).sum(1)

        # convolving with v
        q = q.unsqueeze(1).repeat(1, A, 1) + adv

        # q = Variable(torch.zeros(32, 18, 1)).cuda()
        # adv = Variable(torch.zeros(32, 18, 1)).cuda()

        return pi, v, q, adv, phi


class ACDQNPure(nn.Module):

    def __init__(self):

        super(ACDQNPure, self).__init__()

        # policy estimator
        self.fc_pi = nn.Sequential(
            nn.Linear(512, 18),
        )

        # q-value net
        self.fc_q = nn.Sequential(
            nn.Linear(1024, 1),
            nn.ReLU()
        )

        # embedding

        self.emb_i_f = nn.Embedding(2, 172)
        self.emb_i_v = nn.Embedding(3, 170)
        self.emb_i_h = nn.Embedding(3, 170)

        self.emb_o = nn.Sequential(
            # nn.Dropout(0.1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            # nn.Dropout2d(0.2),
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
            # nn.Dropout(0.1),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def reset_target(self):
        self.fc_pi[0].weight.data.zero_()
        self.fc_pi[0].bias.data.zero_()
        self.fc_q[0].weight.data.zero_()
        self.fc_q[0].bias.data.zero_()

    def forward(self, s, a, beta=None):

        A = a.shape[1]

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        phi = s.detach()
        s = self.cnn_h(s.view(s.size(0), -1))


        # behavioral estimator
        pi = self.fc_pi(s)

        # embedding net
        a_f = a[:, :, 0].long()
        a_v = a[:, :, 1].long()
        a_h = a[:, :, 2].long()

        a = torch.cat([self.emb_i_f(a_f), self.emb_i_v(a_v), self.emb_i_h(a_h)], dim=2)
        batch, seq, channels = a.shape
        a = self.emb_o(a.view(batch * seq, channels))
        a = a.view(batch, seq, 512)

        # aggregation
        s = s.unsqueeze(1)
        s = s.repeat(1, A, 1)
        x = torch.cat([s, a], dim=2)
        batch, seq, channels = x.shape

        # advantage net
        q = self.fc_q(x.view(batch * seq, channels))
        q = q.view(batch, seq, 1)

        v = Variable(torch.zeros(32, 1)).cuda()
        adv = Variable(torch.zeros(32, 18, 1)).cuda()

        return pi, v, q, adv, phi



class ACDQNAout(nn.Module):

    def __init__(self):

        super(ACDQNAout, self).__init__()

        # policy estimator
        self.fc_pi = nn.Sequential(
            nn.Linear(512, 18),
        )

        # q-value net
        self.fc_q = nn.Sequential(
            nn.Linear(512, 18),
            nn.ReLU()
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            # nn.Dropout2d(0.2),
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
            # nn.Dropout(0.1),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def reset_target(self):
        self.fc_pi[0].weight.data.zero_()
        self.fc_pi[0].bias.data.zero_()
        self.fc_q[0].weight.data.zero_()
        self.fc_q[0].bias.data.zero_()

    def forward(self, s, a=None, beta=None):

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        phi = s.detach()
        s = self.cnn_h(s.view(s.size(0), -1))

        # behavioral estimator
        pi = self.fc_pi(s)

        # q-net
        q = self.fc_q(s)
        q = q.unsqueeze(2)

        v = Variable(torch.zeros(32, 1)).cuda()
        adv = Variable(torch.zeros(32, 18, 1)).cuda()

        return pi, v, q, adv, phi





class ACDQNLSTM(nn.Module):

    def __init__(self):

        super(ACDQNLSTM, self).__init__()

        # policy estimator
        self.fc_pi = nn.Sequential(
            nn.Linear(768, 73),
        )

        # q-value net
        self.fc_q = nn.Sequential(
            nn.Linear(1024, 1),
            nn.ReLU()
        )

        # embedding

        self.emb_i_f = nn.Embedding(2, 22)
        self.emb_i_v = nn.Embedding(3, 21)
        self.emb_i_h = nn.Embedding(3, 21)

        # try action post different representation

        self.emb_quarter = nn.Embedding(5, 20)

        self.fc_a = nn.Sequential(
            nn.Linear(84, 256),
            nn.ReLU()
        )

        # recurrent action network

        self.lstm = torch.nn.LSTM(64, 256, 1, batch_first=True)

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            # nn.Dropout2d(0.2),
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
            # nn.Dropout(0.1),
            nn.Linear(10 * 10 * 64, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def reset_target(self):
        self.fc_pi[0].weight.data.zero_()
        self.fc_pi[0].bias.data.zero_()
        self.fc_q[0].weight.data.zero_()
        self.fc_q[0].bias.data.zero_()

    def forward(self, s, a_pre, a_post):

        # state CNN
        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        s = self.cnn_h(s.view(s.size(0), -1))

        # embedding net for pre actions
        a_f = a_pre[:, :, 0]
        a_v = a_pre[:, :, 1]
        a_h = a_pre[:, :, 2]

        a_pre = torch.cat([self.emb_i_f(a_f), self.emb_i_v(a_v), self.emb_i_h(a_h)], dim=2)

        a_pre, (h_0, c_0) = self.lstm(a_pre)
        # embedding net for post actions
        a_pre = a_pre[:, -1, :].squeeze(1)

        a_f = a_post[:, :, 0]
        a_v = a_post[:, :, 1]
        a_h = a_post[:, :, 2]
        quarter = a_post[:, :, 3]

        a_post = torch.cat([self.emb_i_f(a_f), self.emb_i_v(a_v),
                            self.emb_i_h(a_h), self.emb_quarter(quarter)], dim=2)

        a_post = self.fc_a(a_post)
        batch, options, chennels = a_post.shape

        # batch, options, seq_post, channels = a_post.shape
        # a_post = a_post.permute(0, 1, 2, 3).view(options * batch, seq_post, channels)
        #
        # a_f = a_post[:, :, 0]
        # a_v = a_post[:, :, 1]
        # a_h = a_post[:, :, 2]
        #
        # a_post = torch.cat([self.emb_i_f(a_f), self.emb_i_v(a_v), self.emb_i_h(a_h)], dim=2)
        #
        # h_0 = h_0.repeat(1, options, 1)
        # c_0 = c_0.repeat(1, options, 1)
        #
        # a_post, _ = self.lstm(a_post, (h_0, c_0))
        #
        # a_post = a_post[:, -1, :].squeeze(1)

        s = torch.cat([s, a_pre], dim=1)

        # behavioral estimator
        pi = self.fc_pi(s)

        x = s.unsqueeze(1).repeat(1, options, 1)
        x = torch.cat([x, a_post], dim=2)

        q = self.fc_q(x)

        # q = q.view(options, batch, 1)
        # q = q.permute(1, 0, 2)

        return pi, q, s










# class ACDQNHist(nn.Module):
#
#     def __init__(self):
#
#         super(ACDQNHist, self).__init__()
#
#         # policy estimator
#         self.fc_pi = nn.Sequential(
#             # nn.Dropout(0.1),
#             nn.Linear(512, 6),
#         )
#
#         # value net
#         self.fc_v = nn.Sequential(
#             # nn.Dropout(0.1),
#             nn.Linear(512, 1),
#             nn.LeakyReLU()
#         )
#
#         # advantage net
#         # notice that since we do not flip the value in the convolution operation between the two distributions,
#         # adv is expected to learn a reverse representation of the discretized support
#         self.fc_adv = nn.Sequential(
#             # nn.Dropout(0.1),
#             nn.Linear(512, 6),
#         )
#
#         # batch normalization and dropout
#         self.cnn_conv1 = nn.Sequential(
#             nn.Dropout2d(0.2),
#             nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU()
#         )
#
#         self.cnn_conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU()
#         )
#
#         self.cnn_conv3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU()
#         )
#
#         self.cnn_h = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(10 * 10 * 64, 512),
#             nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU()
#         )
#
#         # initialization
#         self.cnn_conv1[1].bias.data.zero_()
#         self.cnn_conv2[1].bias.data.zero_()
#         self.cnn_conv3[1].bias.data.zero_()
#
#     def reset_target(self):
#         self.fc_pi[1].weight.data.zero_()
#         self.fc_pi[1].bias.data.zero_()
#         self.fc_v[1].weight.data.zero_()
#         self.fc_v[1].bias.data.zero_()
#         self.fc_adv[1].weight.data.zero_()
#         self.fc_adv[1].bias.data.zero_()
#
#     def forward(self, s, a):
#
#         # state CNN
#         s = self.cnn_conv1(s)
#         s = self.cnn_conv2(s)
#         s = self.cnn_conv3(s)
#         s = self.cnn_h(s.view(s.size(0), -1))
#
#         # value network
#         v = self.fc_v(s)
#
#         # behavioral estimator
#         pi = self.fc_pi(s)
#
#         # advantage net
#         adv = self.fc_adv(s)
#
#         q = v - 1. / 6 * adv.sum(1)
#         # convolving with v
#         q = q.unsqueeze(1).repeat(1, 6, 1) + adv
#
#         return pi, v, q, adv, s

# class BehavioralDistRNN(nn.Module):
#
#     def __init__(self):
#
#         super(BehavioralDistRNN, self).__init__()
#
#         # history
#         self.rnn = nn.GRU(512, 1024, 1, batch_first=True, dropout=0, bidirectional=False)
#
#         # batch normalization and dropout
#         self.conv1 = nn.Sequential(
#             nn.Dropout2d(0.2),
#             nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
#         )
#
#         self.fc_h = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(10 * 10 * 64, 512),
#             nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
#         )
#
#         self.fc_a = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(8, 512),
#             # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
#         )
#
#         self.fc_a2 = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(8, 512),
#             # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
#         )
#
#         self.fc_r = nn.Linear(8, 512)
#
#         self.fc_z_beta = nn.Linear(1024, 18)
#         if not args.wasserstein:
#             # linear estimators
#             self.fc_z_vs = nn.Linear(1024, args.atoms_short)
#             self.fc_z_vl = nn.Linear(1024, args.atoms_long)
#
#             self.fc_z_qs = nn.Linear(1024, args.atoms_short)
#             self.fc_z_ql = nn.Linear(1024, args.atoms_long)
#         else:
#             self.fc_z_vs = nn.Sequential(
#                 nn.Linear(1024, args.atoms_short),
#                 nn.Softmax(dim=1)
#             )
#             self.fc_z_vl = nn.Sequential(
#                 nn.Linear(1024, args.atoms_long),
#                 nn.Softmax(dim=1)
#             )
#             self.fc_z_qs = nn.Sequential(
#                 nn.Linear(1024, args.atoms_short),
#                 nn.Softmax(dim=1)
#             )
#             self.fc_z_ql = nn.Sequential(
#                 nn.Linear(1024, args.atoms_long),
#                 nn.Softmax(dim=1)
#             )
#
#         # initialization
#         self.conv1[1].bias.data.zero_()
#         self.conv2[1].bias.data.zero_()
#         self.conv3[1].bias.data.zero_()
#         self.fc_r.bias.data.fill_(1)
#
#         a0 = Variable(torch.FloatTensor([1, 0, 0, 0, 1, 0, 0, 1, 0]), requires_grad=False)
#         a0 = a0.unsqueeze(0)
#         self.a0 = a0.unsqueeze(1)
#         r0 = Variable(torch.FloatTensor([0]), requires_grad=False)
#         r0 = r0.unsqueeze(0)
#         self.r0 = r0.unsqueeze(1)
#
#     def forward(self, o, a, r, h, a_next=None):
#
#         batch, seq, channels, height, width = o.shape
#         o = o.view(batch*seq, channels, height, width)
#
#         # state CNN
#         o = F.relu(self.conv1(o))
#         o = F.relu(self.conv2(o))
#         o = F.relu(self.conv3(o))
#         o = F.relu(self.fc_h(o.view(o.size(0), -1)))
#
#         o = o.view(batch, seq, 512)
#
#         # add dumy prev action and prev reward
#
#         a_prev = torch.cat([a[:, -1, :], self.a0], dim=1)
#         r_prev = torch.cat([r[:, -1, :], self.r0], dim=1)
#
#         a_prev = F.leaky_relu(self.fc_a(a_prev))
#         r_prev = F.leaky_relu(self.fc_r(r_prev))
#
#         x = o * a_prev
#         x = x * r_prev
#
#         # calc state with history
#         s, next_h = self.rnn(x, h)
#
#         # value network
#         vs = F.leaky_relu(self.fc_z_vs(s))
#         vl = F.leaky_relu(self.fc_z_vl(s))
#
#         # behavioral estimator
#         beta = F.leaky_relu(self.fc_z_beta(s))
#
#         a = F.leaky_relu(self.fc_a2(a))
#         x = s * a
#
#         # q network
#         qs = F.leaky_relu(self.fc_z_qs(x))
#         ql = F.leaky_relu(self.fc_z_ql(x))
#
#         return vs, vl, beta, qs, ql, next_h
#
#
def wasserstein_metric(support=50, n=1):

    def func(p, q):

        Fp = torch.cumsum(p, dim=1)
        Fq = torch.cumsum(q, dim=1)

        l = torch.sum(torch.abs(Fp - Fq).pow_(n), 1).pow_(1/float(n)) / support
        return l.sum() / p.data.shape[0]

    return func


# self.fc_z_f = nn.Linear(512 + 8, 64)
# self.fc_z_q = nn.Linear(512 + 8 + 64, 1)
# self.fc_z_r = nn.Linear(512 + 8 + 64, 1)
# self.fc_z_p = nn.Linear(512 + 8 + 64, 512)

# self.fc_z_f = nn.Bilinear(512, 8, 64)
# self.fc_z_q = nn.Linear(64, 1)
# self.fc_z_r = nn.Linear(64, 1)
# self.fc_z_p = nn.Linear(64, 512)

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


# class BehavioralRNN(nn.Module):
#
#     def __init__(self):
#
#         super(BehavioralRNN, self).__init__()
#
#         # history
#         self.rnn = nn.GRU(512+3, 256, 1, batch_first=True, dropout=0, bidirectional=False)
#
#         # linear estimators
#         self.fc_z_v = nn.Linear(512 + 256, 1)
#         self.fc_z_beta = nn.Linear(512 + 256, 3)
#         self.fc_z_q = nn.Linear(512 + 256 + 3, 1)
#         self.fc_z_r = nn.Linear(512 + 256 + 3, 1)
#         self.fc_z_p = nn.Linear(512 + 256 + 3, 512)
#
#         # batch normalization and dropout
#         self.conv1 = nn.Sequential(
#             nn.Dropout2d(0.0),
#             nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=(2, 2)),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
#         )
#
#         self.fc_h = nn.Sequential(
#             nn.Dropout(0.0),
#             nn.Linear(10 * 10 * 64, 512),
#             nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
#         )
#
#     def forward(self, o, a, h, a_next=None):
#
#         batch, seq, channels, height, width = o.shape
#         o = o.view(batch*seq, channels, height, width)
#
#         # state CNN
#         o = F.relu(self.conv1(o))
#         o = F.relu(self.conv2(o))
#         o = F.relu(self.conv3(o))
#         o = F.relu(self.fc_h(o.view(o.size(0), -1)))
#
#         o = o.view(batch, seq, 512)
#
#         # calc state history
#         x = torch.cat([o, a], dim=2)
#         s, next_h = self.rnn(x, h.unsqueeze(0))
#
#         s_prev = torch.cat([h.usqueeze(1), s[:, :-1, :]], dim=1)
#
#         x = torch.cat([o, s_prev], dim=2)
#         # value network
#         v = F.relu(self.fc_z_v(x))
#         # behavioral estimator
#         beta = self.fc_z_beta(x)
#
#         if a_next is None:
#             a_next = beta[:, -1, :].sign() * (beta[:, -1, :].abs() > 0.5).float()
#
#         a_next = a_next.unsqueeze(1)
#         a = torch.cat([a, a_next], dim=1)
#         x = torch.cat([o, a, s_prev], dim=2)
#         # reward prediction
#         q = F.relu(self.fc_z_q(x))
#         # reward prediction
#         r = F.relu(self.fc_z_r(x))
#         # state prediction
#         p = F.relu(self.fc_z_p(x))
#
#         return v, q, beta, p, r, next_h