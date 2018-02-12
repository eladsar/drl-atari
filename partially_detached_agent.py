import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn import functional as F

from config import consts, args
from detached_model import BetaNet, CriticNet, ActorNet
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, divide_dataset_by_episodes, \
     SequentialDemonstrationSampler
from agent import Agent
from environment import Env
import random


class DetachedAgent(Agent):

    def __init__(self, load_dataset=True):

        super(DetachedAgent, self).__init__()

        self.meta, self.data = preprocess_demonstrations()

        if load_dataset:
            # demonstration source
            self.meta = divide_dataset_by_episodes(self.meta)

            # datasets
            self.train_dataset = DemonstrationMemory("train", self.meta, self.data)
            self.test_dataset = DemonstrationMemory("test", self.meta, self.data)

            self.train_sampler = DemonstrationBatchSampler(self.train_dataset, train=True)
            self.test_sampler = DemonstrationBatchSampler(self.test_dataset, train=False)

            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            num_workers=args.cpu_workers, pin_memory=True, drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_sampler=self.test_sampler,
                                                           num_workers=args.cpu_workers, pin_memory=True, drop_last=False)

        self.norm = 2
        self.loss_v_beta = torch.nn.MSELoss(size_average=True, reduce=True)
        self.loss_q_pi = torch.nn.MSELoss(size_average=True, reduce=True)

        self.loss_q_beta = torch.nn.MSELoss(size_average=True, reduce=True)

        self.histogram = torch.from_numpy(self.meta['histogram']).float().cuda()
        # weights = self.histogram.max() / self.histogram
        # weights = torch.clamp(weights, 0, 10)
        # weights = 1 - self.histogram

        if self.balance:
            self.loss_beta = torch.nn.CrossEntropyLoss(size_average=True)
        else:
            weights = self.histogram + args.balance_epsilone
            weights = weights.max() / weights
            self.loss_beta = torch.nn.CrossEntropyLoss(size_average=True, weight=weights)

        self.loss_pi = torch.nn.CrossEntropyLoss(reduce=False)

        # actor critic setting
        self.beta_net = BetaNet().cuda()
        self.beta_target = BetaNet().cuda()

        self.pi_net = ActorNet().cuda()
        self.pi_target = ActorNet().cuda()

        self.q_net = CriticNet().cuda()
        self.q_target = CriticNet().cuda()

        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER

        self.optimizer_q_pi = DetachedAgent.set_optimizer(self.q_net.parameters(), 0.0001)  # 0.0002
        self.scheduler_q_pi = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q_pi, self.decay)

        self.optimizer_q_beta = DetachedAgent.set_optimizer(self.beta_net.parameters(), 0.001)  # 0.0002
        self.scheduler_q_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q_beta, self.decay)

        self.optimizer_pi = DetachedAgent.set_optimizer(self.pi_net.parameters(), 0.0002)
        self.scheduler_pi = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi, self.decay)

        self.optimizer_v_beta = DetachedAgent.set_optimizer(self.beta_net.parameters(), 0.001)
        self.scheduler_v_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v_beta, self.decay)

        self.optimizer_beta = DetachedAgent.set_optimizer(self.beta_net.parameters(), 0.01)  # 0.0008
        self.scheduler_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta, self.decay)

        actions = torch.LongTensor(consts.hotvec_matrix).cuda()
        self.actions_matrix = Variable(actions.unsqueeze(0), requires_grad=False)

        self.batch_actions_matrix = self.actions_matrix.repeat(self.batch, 1, 1)

        self.mask_beta = Variable(torch.FloatTensor(consts.behavioral_mask[args.game]), requires_grad=False).cuda()
        self.mask_beta[self.mask_beta == 0] = -float("Inf")
        self.mask_beta[self.mask_beta == 1] = 0
        self.mask_beta_batch = self.mask_beta.repeat(self.batch, 1)

        self.mask_q = Variable(torch.FloatTensor(consts.behavioral_mask[args.game]), requires_grad=False).cuda()
        self.mask_q_batch = self.mask_q.repeat(self.batch, 1)

        self.zero = Variable(torch.zeros(1))

        self.mc = True

    def save_checkpoint(self, path, aux=None):

        state = {'beta_net': self.beta_net.state_dict(),
                 'beta_target': self.beta_target.state_dict(),

                 'pi_net': self.pi_net.state_dict(),
                 'pi_target': self.pi_target.state_dict(),

                 'q_net': self.q_net.state_dict(),
                 'q_target': self.q_target.state_dict(),

                 'optimizer_q_pi': self.optimizer_q_pi.state_dict(),
                 'optimizer_pi': self.optimizer_pi.state_dict(),
                 'optimizer_v_beta': self.optimizer_v_beta.state_dict(),
                 'optimizer_q_beta': self.optimizer_q_beta.state_dict(),
                 'optimizer_beta': self.optimizer_beta.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path)
        self.beta_net.load_state_dict(state['beta_net'])
        self.beta_target.load_state_dict(state['beta_target'])

        self.pi_net.load_state_dict(state['pi_net'])
        self.pi_target.load_state_dict(state['pi_target'])

        self.q_net.load_state_dict(state['q_net'])
        self.q_target.load_state_dict(state['q_target'])

        self.optimizer_q_pi.load_state_dict(state['optimizer_q_pi'])
        self.optimizer_pi.load_state_dict(state['optimizer_pi'])
        self.optimizer_v_beta.load_state_dict(state['optimizer_v_beta'])
        self.optimizer_q_beta.load_state_dict(state['optimizer_q_beta'])
        self.optimizer_beta.load_state_dict(state['optimizer_beta'])

        return state['aux']

    def resume(self, model_path):
        aux = self.load_checkpoint(model_path)
        return aux

    def learn(self, n_interval, n_tot):

        self.beta_net.train()
        self.beta_target.train()

        self.pi_net.train()
        self.pi_target.train()

        self.q_net.train()
        self.q_target.train()


        results = {'n': [], 'loss_v_beta': [], 'loss_q_beta': [], 'loss_beta': [],
                   'loss_v_pi': [], 'loss_q_pi': [], 'loss_pi': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)

            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)

            r = Variable(sample['r'].cuda(async=True).unsqueeze(1), requires_grad=False)
            r_mc = Variable(sample['f'].cuda(async=True).unsqueeze(1), requires_grad=False)

            t = Variable(sample['t'].cuda(async=True).unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True), requires_grad=False)

            a_index_unsqueezed = a_index.unsqueeze(1)

            # Behavioral nets
            beta, q_beta, v_beta, phi = self.beta_net(s)

            # Critic nets
            q_pi = self.q_net(phi)

            # Actor nets:
            pi = self.pi_net(phi)

            # target networks

            # Behavioral target
            beta_target, q_beta_target, v_beta_target, phi_target = self.beta_target(s)

            beta_tag_target, q_beta_tag_target, v_beta_tag_target, phi_tag_target = self.beta_target(s_tag)

            q_pi_target = self.q_target(phi_target)

            pi_tag_target = self.pi_target(phi_tag_target)
            q_pi_tag_target = self.q_target(phi_tag_target)

            # gather q values
            q_pi = q_pi.gather(1, a_index_unsqueezed)
            q_beta = q_beta.gather(1, a_index_unsqueezed)
            # q_beta_target = q_beta_target.gather(1, a_index_unsqueezed)
            q_pi_target = q_pi_target.gather(1, a_index_unsqueezed)

            # behavioral networks
            # V^{\beta} is learned with MC return
            loss_v_beta = self.loss_v_beta(v_beta, r_mc)

            # beta is learned with policy gradient and Q=1
            loss_beta = self.loss_beta(beta, a_index)

            # MC Q-value return to boost the learning of Q^{\pi}
            loss_q_beta = self.loss_q_beta(q_beta, r_mc)

            # critic importance sampling
            # pi_target_sfm = F.softmax(pi_target, 1)
            #
            # cc = torch.clamp(pi_target_sfm / beta_target_sfm, 0, 1)
            # cc = cc.gather(1, a_index_unsqueezed)

            # Critic evaluation

            # evaluate V^{\pi}(s')
            # V^{\pi}(s') = \sum_{a} Q^{\pi}(s',a) \pi(a|s')
            pi_sfm_tag_target = F.softmax(pi_tag_target + self.mask_beta_batch, 1)
            # consider only common actions

            v_tag = (q_pi_tag_target * pi_sfm_tag_target).sum(1)
            v_tag = v_tag.unsqueeze(1)
            v_tag = v_tag.detach()
            # rho = ((1 - cc) * q_beta_target + cc * (r + (self.discount ** k) * (v_tag * (1 - t)))).detach()

            loss_q_pi = self.loss_q_pi(q_pi, r + (self.discount ** k) * (v_tag * (1 - t)))

            # actor importance sampling
            pi_sfm = F.softmax(pi, 1)
            beta_target_sfm = F.softmax(beta_target, 1)
            ca = torch.clamp(pi_sfm / beta_target_sfm, 0, 1)
            ca = ca.gather(1, a_index_unsqueezed)

            # Actor evaluation

            loss_pi = self.loss_pi(pi, a_index)

            # total weight is C^{pi/beta}(s,a) * (Q^{pi}(s,a) - V^{beta}(s))

            # if self.balance:
            #     v_beta_bias = (q_beta * beta_sfm).sum(1).unsqueeze(1)
            # else:
            #     v_beta_bias = v_beta

            weight = (ca * (q_pi_target - v_beta_target)).detach()
            loss_pi = (loss_pi * weight.squeeze(1)).mean()

            # Learning part

            self.optimizer_beta.zero_grad()
            loss_beta.backward(retain_graph=True)
            self.optimizer_beta.step()

            self.optimizer_q_beta.zero_grad()
            loss_q_beta.backward(retain_graph=True)
            self.optimizer_q_beta.step()

            self.optimizer_v_beta.zero_grad()
            loss_v_beta.backward()
            self.optimizer_v_beta.step()

            if not self.mc:

                self.optimizer_pi.zero_grad()
                loss_pi.backward()
                self.optimizer_pi.step()

                self.optimizer_q_pi.zero_grad()
                loss_q_pi.backward()
                self.optimizer_q_pi.step()

            J = (ca * q_pi).squeeze(1)

            R = r_mc.abs().mean()
            Q_n = (q_pi / R).mean()
            # V_n = (v_beta / R).mean()
            LV_n = (loss_v_beta / R ** self.norm).mean() ** (1 / self.norm)
            LQB_n = (loss_q_beta / R ** self.norm).mean() ** (1 / self.norm)
            LQ_n = (loss_q_pi / R ** self.norm).mean() ** (1 / self.norm)
            LPi_n = (J / R).mean()
            LBeta_n = 1 - torch.exp(-loss_beta).mean()

            # add results
            results['loss_beta'].append(LBeta_n.data.cpu().numpy()[0])
            results['loss_v_beta'].append(LV_n.data.cpu().numpy()[0])
            results['loss_q_beta'].append(LQB_n.data.cpu().numpy()[0])
            results['loss_pi'].append(LPi_n.data.cpu().numpy()[0])
            results['loss_v_pi'].append(Q_n.data.cpu().numpy()[0])
            results['loss_q_pi'].append(LQ_n.data.cpu().numpy()[0])
            results['n'].append(n)

            if not n % self.update_target_interval:
                self.q_target.load_state_dict(self.q_net.state_dict())
                self.pi_target.load_state_dict(self.pi_net.state_dict())
                self.beta_target.load_state_dict(self.beta_net.state_dict())


            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step()

            # start training the model with behavioral initialization
            if (n+1) == self.update_target_interval * 8:
                self.mc = False
                self.q_target.fc_critic.load_state_dict(self.beta_net.fc_q.state_dict())
                self.q_net.fc_critic.load_state_dict(self.beta_net.fc_q.state_dict())
                self.pi_net.fc_actor.load_state_dict(self.beta_net.fc_pi.state_dict())
                self.pi_target.fc_actor.load_state_dict(self.beta_net.fc_pi.state_dict())

            if not (n+1) % n_interval:

                yield results
                self.beta_net.train()
                self.beta_target.train()

                self.pi_net.train()
                self.pi_target.train()

                self.q_net.train()
                self.q_target.train()

                results = {key: [] for key in results}

    def test(self, n_interval, n_tot):

        self.beta_net.eval()
        self.beta_target.eval()

        self.pi_net.eval()
        self.pi_target.eval()

        self.q_net.eval()
        self.q_target.eval()

        results = {'n': [], 'act_diff': [], 'a_agent': [], 'a_player': [],
                   'loss_v_beta': [], 'loss_q_beta': [], 'loss_beta': [],
                   'loss_v_pi': [], 'loss_q_pi': [], 'loss_pi': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)

            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)

            r = Variable(sample['r'].cuda(async=True).unsqueeze(1), requires_grad=False)
            r_mc = Variable(sample['f'].cuda(async=True).unsqueeze(1), requires_grad=False)

            t = Variable(sample['t'].cuda(async=True).unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True), requires_grad=False)

            a_index_unsqueezed = a_index.unsqueeze(1)

            # Behavioral nets
            beta, q_beta, v_beta, phi = self.beta_net(s)

            # Critic nets
            q_pi = self.q_net(phi)

            # Actor nets:
            pi = self.pi_net(phi)

            # target networks

            # Behavioral target
            beta_target, q_beta_target, v_beta_target, phi_target = self.beta_target(s)

            beta_tag_target, q_beta_tag_target, v_beta_tag_target, phi_tag_target = self.beta_target(s_tag)

            q_pi_target = self.q_target(phi_target)

            pi_tag_target = self.pi_target(phi_tag_target)
            q_pi_tag_target = self.q_target(phi_tag_target)

            # gather q values
            q_pi = q_pi.gather(1, a_index_unsqueezed)
            q_beta = q_beta.gather(1, a_index_unsqueezed)
            # q_beta_target = q_beta_target.gather(1, a_index_unsqueezed)
            q_pi_target = q_pi_target.gather(1, a_index_unsqueezed)

            # behavioral networks
            # V^{\beta} is learned with MC return
            loss_v_beta = self.loss_v_beta(v_beta, r_mc)

            # beta is learned with policy gradient and Q=1
            loss_beta = self.loss_beta(beta, a_index)

            # MC Q-value return to boost the learning of Q^{\pi}
            loss_q_beta = self.loss_q_beta(q_beta, r_mc)

            # critic importance sampling
            # pi_target_sfm = F.softmax(pi_target, 1)
            #
            # cc = torch.clamp(pi_target_sfm / beta_target_sfm, 0, 1)
            # cc = cc.gather(1, a_index_unsqueezed)

            # Critic evaluation

            # evaluate V^{\pi}(s')
            # V^{\pi}(s') = \sum_{a} Q^{\pi}(s',a) \pi(a|s')
            pi_sfm_tag_target = F.softmax(pi_tag_target + self.mask_beta_batch, 1)
            # consider only common actions

            v_tag = (q_pi_tag_target * pi_sfm_tag_target).sum(1)
            v_tag = v_tag.unsqueeze(1)
            v_tag = v_tag.detach()
            # rho = ((1 - cc) * q_beta_target + cc * (r + (self.discount ** k) * (v_tag * (1 - t)))).detach()

            loss_q_pi = self.loss_q_pi(q_pi, r + (self.discount ** k) * (v_tag * (1 - t)))

            # actor importance sampling
            pi_sfm = F.softmax(pi, 1)
            beta_target_sfm = F.softmax(beta_target, 1)
            ca = torch.clamp(pi_sfm / beta_target_sfm, 0, 1)
            ca = ca.gather(1, a_index_unsqueezed)

            # Actor evaluation

            loss_pi = self.loss_pi(pi, a_index)

            # total weight is C^{pi/beta}(s,a) * (Q^{pi}(s,a) - V^{beta}(s))

            # if self.balance:
            #     v_beta_bias = (q_beta * beta_sfm).sum(1).unsqueeze(1)
            # else:
            #     v_beta_bias = v_beta

            weight = (ca * (q_pi_target - v_beta_target)).detach()
            loss_pi = (loss_pi * weight.squeeze(1)).mean()

            # collect actions statistics
            a_index_np = a_index.data.cpu().numpy()

            _, beta_index = beta.data.cpu().max(1)
            beta_index = beta_index.numpy()
            act_diff = (a_index_np != beta_index).astype(np.int)

            # add results
            results['act_diff'].append(act_diff)
            results['a_agent'].append(beta_index)
            results['a_player'].append(a_index_np)

            J = (ca * q_pi).squeeze(1)

            R = r_mc.abs().mean()
            Q_n = (q_pi / R).mean()
            # V_n = (v_beta / R).mean()
            LV_n = (loss_v_beta / R ** self.norm).mean() ** (1 / self.norm)
            LQB_n = (loss_q_beta / R ** self.norm).mean() ** (1 / self.norm)
            LQ_n = (loss_q_pi / R ** self.norm).mean() ** (1 / self.norm)
            LPi_n = (J / R).mean()
            LBeta_n = 1 - torch.exp(-loss_beta).mean()

            # add results
            results['loss_beta'].append(LBeta_n.data.cpu().numpy()[0])
            results['loss_v_beta'].append(LV_n.data.cpu().numpy()[0])
            results['loss_q_beta'].append(LQB_n.data.cpu().numpy()[0])
            results['loss_pi'].append(LPi_n.data.cpu().numpy()[0])
            results['loss_v_pi'].append(Q_n.data.cpu().numpy()[0])
            results['loss_q_pi'].append(LQ_n.data.cpu().numpy()[0])
            results['n'].append(n)

            if not (n+1) % n_interval:

                results['s'] = s.data.cpu()
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_agent'] = np.concatenate(results['a_agent'])
                results['a_player'] = np.concatenate(results['a_player'])
                yield results
                self.beta_net.eval()
                self.beta_target.eval()

                self.pi_net.eval()
                self.pi_target.eval()

                self.q_net.eval()
                self.q_target.eval()

                results = {key: [] for key in results}

    def play(self, n_tot, action_offset, player):

        self.beta_net.eval()
        self.beta_target.eval()

        self.pi_net.eval()
        self.pi_target.eval()

        self.q_net.eval()
        self.q_target.eval()

        env = Env(action_offset)

        n_human = 90

        episodes = list(self.data.keys())
        random.shuffle(episodes)
        humans_trajectories = iter(episodes)

        for i in range(n_tot):

            env.reset()
            trajectory = self.data[next(humans_trajectories)]
            choices = np.arange(self.global_action_space, dtype=np.int)
            random_choices = self.mask_q.data.cpu().numpy()
            random_choices = random_choices / random_choices.sum()

            j = 0

            while not env.t:

                s = Variable(env.s.cuda(), requires_grad=False)

                if player is 'beta':
                    pi, _, _, _ = self.beta_net(s)
                    pi = pi.squeeze(0)
                    self.greedy = False

                elif player is 'q_b':
                    _, pi, _, _ = self.beta_net(s)
                    pi = pi.squeeze(0)
                    self.greedy = True

                elif player is 'pi':
                    _, _, _, phi = self.beta_net(s)
                    pi = self.pi_net(phi)
                    pi = pi.squeeze(0)
                    self.greedy = False

                elif player is 'q_pi':
                    _, _, _, phi = self.beta_net(s)
                    pi = self.q_net(phi)
                    pi = pi.squeeze(0)
                    self.greedy = True

                else:
                    raise NotImplementedError

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:
                    eps = np.random.rand()
                    # eps = 1
                    # a = np.random.choice(choices)
                    if self.greedy:
                        if eps > 0.01:
                            a = (pi*self.mask_q).data.cpu().numpy()
                            a = np.argmax(a)
                        else:
                            a = np.random.choice(choices, p=random_choices)
                    else:
                        a = F.softmax(pi + self.mask_beta, dim=0).data.cpu().numpy()
                        a = np.random.choice(choices, p=a)

                env.step(a)

                j += 1

            yield {'score': env.score,
                   'frames': j}

        raise StopIteration


    def play_episode(self, n_tot):

        self.beta_net.train()
        self.beta_target.train()

        self.pi_net.train()
        self.pi_target.train()

        self.q_net.train()
        self.q_target.train()


        env = Env()

        n_human = 120
        humans_trajectories = iter(self.data)
        softmax = torch.nn.Softmax()

        for i in range(n_tot):

            env.reset()
            observation = next(humans_trajectories)
            trajectory = self.data[observation]
            choices = np.arange(self.global_action_space, dtype=np.int)
            mask = Variable(torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
                             requires_grad=False).cuda()
            j = 0
            temp = 1

            while not env.t:

                s = Variable(env.s.cuda(), requires_grad=False)

                beta, phi = self.beta_net(s)
                pi, _ = self.pi_net(s)
                q, _ = self.q_net(s)
                vb, _ = self.vb_net(s)

                pi = beta.squeeze(0)
                self.greedy = False

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:
                    # eps = np.random.rand()
                    eps = 1
                    # a = np.random.choice(choices)
                    if self.greedy and eps > 0.01:
                        a = pi.data.cpu().numpy()
                        a = np.argmax(a)
                    else:
                        a = softmax(pi/temp).data.cpu().numpy()
                        a = np.random.choice(choices, p=a)

                q = q[0, a]
                q = q.squeeze(0)

                env.step(a)

                yield {'o': env.s.cpu().numpy(),
                       'v': vb.squeeze(0).data.cpu().numpy(),
                       'vb': vb.squeeze(0).data.cpu().numpy(),
                       'qb': q.squeeze(0).data.cpu().numpy(),
                       # 's': x[0, :512].data.cpu().numpy(),
                       'score': env.score,
                       'beta': pi.data.cpu().numpy(),
                       'phi': phi.squeeze(0).data.cpu().numpy(),
                       'q': q.squeeze(0).data.cpu().numpy()}

                j += 1

        raise StopIteration