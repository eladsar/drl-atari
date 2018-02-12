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
from detached_model import DQN, DVN, DPiN
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
            weights = self.histogram + 0.01
            weights = weights.max() / weights
            self.loss_beta = torch.nn.CrossEntropyLoss(size_average=True, weight=weights)

        self.loss_pi = torch.nn.CrossEntropyLoss(reduce=False)

        # actor critic setting
        self.beta_net = DPiN().cuda()
        self.beta_target = DPiN().cuda()

        self.pi_net = DPiN().cuda()
        self.pi_target = DPiN().cuda()

        self.vb_net = DVN().cuda()
        self.vb_target = DVN().cuda()

        self.qb_net = DQN().cuda()
        self.qb_target = DQN().cuda()

        self.q_net = DQN().cuda()
        self.q_target = DQN().cuda()

        self.best_behavioral = np.inf
        self.best_vb = np.inf
        self.best_qb = np.inf

        self.q_target.reset_target()
        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER

        self.optimizer_q_pi = DetachedAgent.set_optimizer(self.q_net.parameters(), 0.0001)  # 0.0002
        self.scheduler_q_pi = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q_pi, self.decay)

        self.optimizer_q_beta = DetachedAgent.set_optimizer(self.qb_net.parameters(), 0.0001)  # 0.0002
        self.scheduler_q_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q_beta, self.decay)

        self.optimizer_pi = DetachedAgent.set_optimizer(self.pi_net.parameters(), 0.0002)
        self.scheduler_pi = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi, self.decay)

        self.optimizer_v_beta = DetachedAgent.set_optimizer(self.vb_net.parameters(), 0.0001)
        self.scheduler_v_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v_beta, self.decay)

        self.optimizer_beta = DetachedAgent.set_optimizer(self.beta_net.parameters(), 0.0006)  # 0.0008
        self.scheduler_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta, self.decay)

        actions = torch.LongTensor(consts.hotvec_matrix).cuda()
        self.actions_matrix = Variable(actions.unsqueeze(0), requires_grad=False)

        self.batch_actions_matrix = self.actions_matrix.repeat(self.batch, 1, 1)

        self.mask_beta = Variable(torch.FloatTensor(consts.behavioral_mask[args.game]), requires_grad=False).cuda()

        self.zero = Variable(torch.zeros(1))

        self.mc = True

    def save_checkpoint(self, path, aux=None):

        state = {'beta_net': self.beta_net.state_dict(),
                 'pi_net': self.pi_net.state_dict(),
                 'vb_net': self.vb_net.state_dict(),
                 'q_net': self.q_net.state_dict(),
                 'q_target': self.q_target.state_dict(),
                 'qb_net': self.qb_net.state_dict(),
                 'qb_target': self.qb_target.state_dict(),
                 'beta_target': self.beta_target.state_dict(),
                 'vb_target': self.vb_target.state_dict(),
                 'best_behavioral': self.best_behavioral,
                 'best_vb': self.best_vb,
                 'best_qb': self.best_qb,
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
        self.pi_net.load_state_dict(state['pi_net'])
        self.vb_net.load_state_dict(state['vb_net'])
        self.q_net.load_state_dict(state['q_net'])
        self.q_target.load_state_dict(state['q_target'])
        self.qb_net.load_state_dict(state['qb_net'])
        self.qb_target.load_state_dict(state['qb_target'])
        self.beta_target.load_state_dict(state['beta_target'])
        self.vb_target.load_state_dict(state['vb_target'])
        self.best_behavioral = state['best_behavioral']
        self.best_vb = state['best_vb']
        self.best_qb = state['best_qb']
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
        self.pi_net.train()
        self.vb_net.train()
        self.q_net.train()
        self.qb_net.train()
        self.qb_target.train()
        self.q_target.train()
        self.beta_target.train()
        self.vb_target.train()

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

            # Behavioral nets
            beta, _ = self.beta_net(s)
            v_beta, _ = self.vb_net(s)
            q_beta, _ = self.qb_net(s)

            # Critic nets
            q_pi, _ = self.q_net(s)
            pi_target_tag, _ = self.pi_target(s_tag)
            beta_target_tag, _ = self.beta_target(s_tag)
            q_pi_tag_target, _ = self.q_target(s_tag)

            # Actor nets:
            # V(s) bias:
            v_beta_target, _ = self.vb_target(s)
            beta_target, _ = self.beta_target(s)
            q_beta_target, _ = self.qb_target(s)
            pi, _ = self.pi_net(s)

            q_pi = q_pi.gather(1, a_index.unsqueeze(1))
            q_beta = q_beta.gather(1, a_index.unsqueeze(1))

            # behavioral networks
            # V^{\beta} is learned with MC return
            loss_v_beta = self.loss_v_beta(v_beta, r_mc)

            # beta is learned with policy gradient and Q=1
            loss_beta = self.loss_beta(beta, a_index)

            # MC Q-value return to boost the learning of Q^{\pi}
            loss_q_beta = self.loss_q_beta(q_beta, r_mc)

            # importance sampling
            beta_sfm = F.softmax(beta_target, 1)
            pi_sfm = F.softmax(pi, 1)

            c = torch.clamp(pi_sfm/beta_sfm, 0, 1)
            c = c.gather(1, a_index.unsqueeze(1))

            # Critic evaluation

            # evaluate V^{\pi}(s')
            # V^{\pi}(s') = \sum_{a} Q^{\pi}(s',a) \pi(a|s')
            beta_sfm_tag = F.softmax(beta_target_tag, 1)
            pi_sfm_tag = F.softmax(pi_target_tag, 1)
            # consider only common actions
            mask_b = (beta_sfm_tag > self.behavioral_threshold).float()

            v_tag = (q_pi_tag_target * mask_b * pi_sfm_tag).sum(1)
            v_tag = v_tag.unsqueeze(1)
            v_tag = v_tag.detach()
            q_mc = q_beta_target.gather(1, a_index.unsqueeze(1))
            rho = ((1 - c) * q_mc + r + (self.discount ** k) * (v_tag * (1 - t))).detach()

            loss_q_pi = self.loss_q_pi(q_pi, rho)

            # Actor evaluation

            loss_pi = self.loss_pi(pi, a_index)

            # total weight is C^{pi/beta}(s,a) * (Q^{pi}(s,a) - V^{beta}(s))
            if self.balance:
                v_beta_target = (q_beta_target * beta_sfm).sum(1).unsqueeze(1)

            weight = (c * (q_pi - v_beta_target)).detach()
            loss_pi = (loss_pi * weight.squeeze(1)).mean()

            # Learning part

            self.optimizer_beta.zero_grad()
            loss_beta.backward()
            self.optimizer_beta.step()

            self.optimizer_q_beta.zero_grad()
            loss_q_beta.backward()
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

            J = (c * q_pi).squeeze(1)

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

            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step()

            # start training the model with behavioral initialization
            if (n+1) == self.update_n_steps_interval * 4:
                self.mc = False
                self.q_target.load_state_dict(self.qb_net.state_dict())
                self.q_net.load_state_dict(self.qb_net.state_dict())
                self.pi_net.load_state_dict(self.beta_net.state_dict())
                self.pi_target.load_state_dict(self.beta_net.state_dict())

            if not (n+1) % n_interval:

                yield results
                self.beta_net.train()
                self.pi_net.train()
                self.vb_net.train()
                self.q_net.train()
                self.qb_net.train()
                self.qb_target.train()
                self.q_target.train()
                self.beta_target.train()
                self.vb_target.train()
                results = {key: [] for key in results}

    def test(self, n_interval, n_tot):

        self.beta_net.eval()
        self.pi_net.eval()
        self.vb_net.eval()
        self.q_net.eval()
        self.q_target.eval()
        self.beta_target.eval()
        self.vb_target.eval()
        self.qb_net.eval()
        self.qb_target.eval()

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

            # Behavioral nets
            beta, _ = self.beta_net(s)
            v_beta, _ = self.vb_net(s)
            q_beta, _ = self.qb_net(s)

            # Critic nets
            q_pi, _ = self.q_net(s)
            pi_target_tag, _ = self.pi_target(s_tag)
            beta_target_tag, _ = self.beta_target(s_tag)
            q_pi_tag_target, _ = self.q_target(s_tag)

            # Actor nets:
            # V(s) bias:
            v_beta_target, _ = self.vb_target(s)
            beta_target, _ = self.beta_target(s)
            q_beta_target, _ = self.qb_target(s)
            pi, _ = self.pi_net(s)

            q_pi = q_pi.gather(1, a_index.unsqueeze(1))
            q_beta = q_beta.gather(1, a_index.unsqueeze(1))

            # behavioral networks
            # V^{\beta} is learned with MC return
            loss_v_beta = self.loss_v_beta(v_beta, r_mc)

            # beta is learned with policy gradient and Q=1
            loss_beta = self.loss_beta(beta, a_index)

            # MC Q-value return to boost the learning of Q^{\pi}
            loss_q_beta = self.loss_q_beta(q_beta, r_mc)

            # importance sampling
            beta_sfm = F.softmax(beta_target, 1)
            pi_sfm = F.softmax(pi, 1)

            c = torch.clamp(pi_sfm/beta_sfm, 0, 1)
            c = c.gather(1, a_index.unsqueeze(1))

            # Critic evaluation

            # evaluate V^{\pi}(s')
            # V^{\pi}(s') = \sum_{a} Q^{\pi}(s',a) \pi(a|s')
            beta_sfm_tag = F.softmax(beta_target_tag, 1)
            pi_sfm_tag = F.softmax(pi_target_tag, 1)
            # consider only common actions
            mask_b = (beta_sfm_tag > self.behavioral_threshold).float()

            v_tag = (q_pi_tag_target * mask_b * pi_sfm_tag).sum(1)
            v_tag = v_tag.unsqueeze(1)
            v_tag = v_tag.detach()
            q_mc = q_beta_target.gather(1, a_index.unsqueeze(1))
            rho = ((1 - c) * q_mc + r + (self.discount ** k) * (v_tag * (1 - t))).detach()

            loss_q_pi = self.loss_q_pi(q_pi, rho)

            # Actor evaluation

            loss_pi = self.loss_pi(pi, a_index)

            # total weight is C^{pi/beta}(s,a) * (Q^{pi}(s,a) - V^{beta}(s))
            if self.balance:
                v_beta_target = (q_beta_target * beta_sfm).sum(1).unsqueeze(1)

            weight = (c * (q_pi - v_beta_target)).detach()
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

            J = (c * q_pi).squeeze(1)

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

                # update best behavioral policy if needed
                LBeta_avg = np.array(results['loss_beta']).mean()
                if LBeta_avg < self.best_behavioral:
                    self.beta_target.load_state_dict(self.beta_net.state_dict())
                    self.best_behavioral = LBeta_avg

                LV_avg = np.array(results['loss_v_beta']).mean()
                if LV_avg < self.best_vb:
                    self.vb_target.load_state_dict(self.vb_net.state_dict())
                    self.best_vb = LV_avg

                LQB_avg = np.array(results['loss_q_beta']).mean()
                if LQB_avg < self.best_qb:
                    self.qb_target.load_state_dict(self.qb_net.state_dict())
                    self.best_qb = LQB_avg

                results['s'] = s.data.cpu()
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_agent'] = np.concatenate(results['a_agent'])
                results['a_player'] = np.concatenate(results['a_player'])
                yield results
                self.beta_net.eval()
                self.pi_net.eval()
                self.vb_net.eval()
                self.q_net.eval()
                self.q_target.eval()
                self.qb_target.eval()
                self.beta_target.eval()
                self.vb_target.eval()
                self.qb_net.eval()
                results = {key: [] for key in results}

    def play(self, n_tot, action_offset, player):

        self.beta_net.eval()
        self.pi_net.eval()
        self.vb_net.eval()
        self.q_net.eval()
        self.q_target.eval()
        self.beta_target.eval()
        self.vb_target.eval()
        self.qb_net.eval()
        self.qb_target.eval()

        env = Env(action_offset)

        n_human = 90

        episodes = list(self.data.keys())
        random.shuffle(episodes)
        humans_trajectories = iter(episodes)

        for i in range(n_tot):

            env.reset()
            trajectory = self.data[next(humans_trajectories)]
            choices = np.arange(self.global_action_space, dtype=np.int)

            j = 0

            while not env.t:

                s = Variable(env.s.cuda(), requires_grad=False)

                if player is 'beta':
                    pi, _ = self.beta_net(s)
                    pi = pi.squeeze(0) * self.mask_beta
                    self.greedy = False

                elif player is 'q_b':
                    pi, _ = self.qb_net(s)
                    pi = pi.squeeze(0) * self.mask_beta
                    self.greedy = True

                elif player is 'pi':
                    pi, _ = self.pi_net(s)
                    pi = pi.squeeze(0) * self.mask_beta
                    self.greedy = False

                elif player is 'q_pi':
                    pi, _ = self.q_net(s)
                    pi = pi.squeeze(0) * self.mask_beta
                    self.greedy = True

                else:
                    raise NotImplementedError

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:
                    eps = np.random.rand()
                    # eps = 1
                    # a = np.random.choice(choices)
                    if self.greedy and eps > 0.01:
                        a = pi.data.cpu().numpy()
                        a = np.argmax(a)
                    else:
                        a = F.softmax(pi, dim=0).data.cpu().numpy()
                        a = np.random.choice(choices, p=a)

                env.step(a)

                j += 1

            yield {'score': env.score,
                   'frames': j}

        raise StopIteration


    def play_episode(self, n_tot):

        self.beta_net.eval()
        self.pi_net.eval()
        self.vb_net.eval()
        self.q_net.eval()
        self.q_target.eval()
        self.beta_target.eval()
        self.vb_target.eval()
        self.qb_net.eval()

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
                    if self.greedy and eps > 0.025:
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


        # if self.mc:
        #
        #     # MC return to boost the learning of Q^{\pi}
        #     loss_q_pi = self.loss_q_pi(q_pi, r_mc)
        # else:
        #
        #     # evaluate V^{\pi}(s')
        #     # V^{\pi}(s') = \sum_{a} Q^{\pi}(s',a) \pi(a|s')
        #     pi_target_tag, _ = self.pi_target(s_tag)
        #     beta_target_tag, _ = self.beta_target(s_tag)
        #     beta_sfm_tag = F.softmax(beta_target_tag, 1)
        #     pi_sfm_tag = F.softmax(pi_target_tag, 1)
        #     # consider only common actions
        #     mask_b = (beta_sfm_tag > self.behavioral_threshold).float()
        #     q_pi_tag_target, _ = self.q_target(s_tag)
        #
        #     v_tag = (q_pi_tag_target * mask_b * pi_sfm_tag).sum(1)
        #     v_tag = v_tag.unsqueeze(1)
        #     v_tag = v_tag.detach()
        #
        #     loss_q_pi = self.loss_q_pi(q_pi, r + (self.discount ** k) * (v_tag * (1 - t)))