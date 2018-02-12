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
from model import ACDQNLSTM
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationSampler
from agent import Agent
from environment import Env


class ACDQNLSTMAgent(Agent):

    def __init__(self, load_dataset=True):

        super(ACDQNLSTMAgent, self).__init__()

        self.meta, self.data = preprocess_demonstrations()

        if load_dataset:
            # demonstration source
            self.meta = divide_dataset(self.meta)

            # datasets
            self.train_dataset = DemonstrationMemory("train", self.meta, self.data)
            self.test_dataset = DemonstrationMemory("test", self.meta, self.data)

            self.train_sampler = DemonstrationBatchSampler(self.train_dataset, train=True)
            self.test_sampler = DemonstrationBatchSampler(self.test_dataset, train=False)

            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            num_workers=args.cpu_workers, pin_memory=True, drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_sampler=self.test_sampler,
                                                           num_workers=args.cpu_workers, pin_memory=True, drop_last=False)

        self.loss_v_beta = torch.nn.L1Loss(size_average=True, reduce=True)
        self.loss_q_beta = torch.nn.L1Loss(size_average=True, reduce=True)

        self.loss_v_pi = torch.nn.L1Loss(size_average=True, reduce=True)
        self.loss_q_pi = torch.nn.L1Loss(size_average=True, reduce=True)

        self.loss_p = torch.nn.L1Loss(size_average=True, reduce=True)

        self.histogram = torch.from_numpy(self.meta['histogram']).float()
        weights = self.histogram.max() / self.histogram
        weights = torch.clamp(weights, 0, 10).cuda()

        self.loss_beta = torch.nn.CrossEntropyLoss(size_average=True)
        self.loss_pi = torch.nn.CrossEntropyLoss(reduce=False)

        # actor critic setting

        self.model_b_single = ACDQNLSTM().cuda()
        self.model_single = ACDQNLSTM().cuda()
        self.target_single = ACDQNLSTM().cuda()

        if self.parallel:
            self.model_b = torch.nn.DataParallel(self.model_b_single)
            self.model = torch.nn.DataParallel(self.model_single)
            self.target = torch.nn.DataParallel(self.target_single)
        else:
            self.model_b = self.model_b_single
            self.model = self.model_single
            self.target = self.target_single

        self.target_single.reset_target()
        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER


        self.optimizer_q_pi = ACDQNLSTMAgent.set_optimizer(self.model.parameters(), 0.0002)
        self.scheduler_q_pi = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q_pi, self.decay)

        self.optimizer_pi = ACDQNLSTMAgent.set_optimizer(self.model.parameters(), 0.0002)
        self.scheduler_pi = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi, self.decay)

        self.optimizer_q_beta = ACDQNLSTMAgent.set_optimizer(self.model_b.parameters(), 0.0002)
        self.scheduler_q_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q_beta, self.decay)

        self.optimizer_beta = ACDQNLSTMAgent.set_optimizer(self.model_b.parameters(), 0.0008)
        self.scheduler_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta, self.decay)

        actions = torch.LongTensor(consts.hotvec_matrix).cuda()
        self.actions_matrix = Variable(actions.unsqueeze(0), requires_grad=False)

        self.batch_actions_matrix = self.actions_matrix.repeat(self.batch, 1, 1)

        self.batch_range = np.arange(self.batch)
        self.zero = Variable(torch.zeros(1))
        self.a_post_mat = Variable(torch.from_numpy(consts.a_post_mat).long(), requires_grad=False).cuda()
        self.a_post_mat = self.a_post_mat.unsqueeze(0).repeat(self.batch, 1, 1)

    def save_checkpoint(self, path, aux=None):

        state = {'model_b': self.model_b.state_dict(),
                 'model': self.model.state_dict(),
                 'target': self.target.state_dict(),
                 'optimizer_q_pi': self.optimizer_q_pi.state_dict(),
                 'optimizer_pi': self.optimizer_pi.state_dict(),
                 'optimizer_q_beta': self.optimizer_q_beta.state_dict(),
                 'optimizer_beta': self.optimizer_beta.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path)
        self.model_b.load_state_dict(state['model_b'])
        self.model.load_state_dict(state['model'])
        self.target.load_state_dict(state['target'])
        self.optimizer_q_pi.load_state_dict(state['optimizer_q_pi'])
        self.optimizer_pi.load_state_dict(state['optimizer_pi'])
        self.optimizer_q_beta.load_state_dict(state['optimizer_q_beta'])
        self.optimizer_beta.load_state_dict(state['optimizer_beta'])

        return state['aux']

    def resume(self, model_path):

        aux = self.load_checkpoint(model_path)
        # self.update_target()
        return aux

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def learn(self, n_interval, n_tot):

        self.model_b.train()
        self.model.train()
        self.target.eval()

        results = {'n': [], 'loss_v_beta': [], 'loss_q_beta': [], 'loss_beta': [],
                   'loss_v_pi': [], 'loss_q_pi': [], 'loss_pi': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)

            a_pre = Variable(sample['horizon_pre'].cuda(async=True), requires_grad=False)
            a_pre_tag = Variable(sample['horizon_pre_tag'].cuda(async=True), requires_grad=False)

            a_index = Variable(sample['matched_option'].cuda(async=True), requires_grad=False)

            r = Variable(sample['r'].cuda(async=True).unsqueeze(1), requires_grad=False)
            r_mc = Variable(sample['f'].cuda(async=True).unsqueeze(1), requires_grad=False)

            t = Variable(sample['t'].cuda(async=True).unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True), requires_grad=False)

            beta, q_beta, _, = self.model_b(s, a_pre, self.a_post_mat)

            q_beta = q_beta.squeeze(2)
            q_beta = q_beta.gather(1, a_index)

            loss_beta = self.loss_beta(beta, a_index.squeeze(1))
            loss_v_beta = r_mc.abs().mean()
            loss_q_beta = self.loss_q_beta(q_beta, r_mc)

            pi, q_pi, _ = self.model(s, a_pre, self.a_post_mat)
            pi_tag, q_tag, _ = self.target(s_tag, a_pre_tag, self.a_post_mat)

            q_pi = q_pi.squeeze(2)
            q_pi = q_pi.gather(1, a_index)

            # ignore negative q:
            if self.double_q:
                _,  q_tag_model, _ = self.model(s_tag, a_pre_tag, self.a_post_mat)
                _, a_tag = F.relu(q_tag_model).max(1)
                q_max = q_tag.gather(1, a_tag.unsqueeze(1))
                q_max = F.relu(q_max).squeeze(1)
            else:
                q_max, _ = F.relu(q_tag).max(1)

            q_max = q_max.detach()

            loss_v_pi = (r + (self.discount ** k) * (q_max * (1 - t))).abs().mean()
            loss_q_pi = self.loss_q_pi(q_pi, r + (self.discount ** k) * (q_max * (1 - t)))
            loss_pi = self.loss_pi(pi, a_index.squeeze(1))

            beta_sfm = F.softmax(beta, 1)
            pi_sfm = F.softmax(pi, 1)
            c = torch.clamp(pi_sfm/beta_sfm, 0, 1)

            c = c.gather(1, a_index)
            weight = (c * q_pi).detach()
            loss_pi = (loss_pi * weight.squeeze(1)).sum()

            self.optimizer_beta.zero_grad()
            loss_beta.backward(retain_graph=True)
            self.optimizer_beta.step()

            self.optimizer_q_beta.zero_grad()
            loss_q_beta.backward()
            self.optimizer_q_beta.step()

            self.optimizer_pi.zero_grad()
            loss_pi.backward(retain_graph=True)
            self.optimizer_pi.step()

            # self.optimizer_v_pi.zero_grad()
            # loss_v_pi.backward()
            # self.optimizer_v_pi.step()

            self.optimizer_q_pi.zero_grad()
            loss_q_pi.backward()
            self.optimizer_q_pi.step()

            R = (r_mc ** 1).mean()

            # add results
            results['loss_beta'].append(loss_beta.data.cpu().numpy()[0])
            results['loss_v_beta'].append((loss_v_beta / R).data.cpu().numpy()[0])
            results['loss_q_beta'].append((loss_q_beta / R).data.cpu().numpy()[0])
            # results['loss_q_beta'].append(loss_p.data.cpu().numpy()[0])
            results['loss_pi'].append(loss_pi.data.cpu().numpy()[0])
            results['loss_v_pi'].append((loss_v_pi / R).data.cpu().numpy()[0])
            results['loss_q_pi'].append((loss_q_pi / R).data.cpu().numpy()[0])
            results['n'].append(n)


            if not n % self.update_target_interval:
                self.update_target()

            # if an index is rolled more than once during update_memory_interval period, only the last occurance affect the
            if not (n+1) % self.update_memory_interval and self.prioritized_replay:
                self.train_dataset.update_probabilities()

            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step()

            # start training the model with behavioral initialization
            if (n+1) == self.update_n_steps_interval:
                self.target_single.reset_target()
                self.model.load_state_dict(self.model_b.state_dict())

            if not (n+1) % n_interval:
                yield results
                self.model_b.train()
                self.model.train()
                self.target.eval()
                results = {key: [] for key in results}

    def test(self, n_interval, n_tot):

        self.model.eval()
        self.target.eval()
        self.model_b.eval()

        results = {'n': [], 'act_diff': [], 'a_agent': [], 'a_player': [],
                   'loss_v_beta': [], 'loss_q_beta': [], 'loss_beta': [],
                   'loss_v_pi': [], 'loss_q_pi': [], 'loss_pi': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)

            a_pre = Variable(sample['horizon_pre'].cuda(async=True), requires_grad=False)
            a_pre_tag = Variable(sample['horizon_pre_tag'].cuda(async=True), requires_grad=False)

            a_index = Variable(sample['matched_option'].cuda(async=True), requires_grad=False)

            r = Variable(sample['r'].cuda(async=True).unsqueeze(1), requires_grad=False)
            r_mc = Variable(sample['f'].cuda(async=True).unsqueeze(1), requires_grad=False)

            t = Variable(sample['t'].cuda(async=True).unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True), requires_grad=False)

            beta, q_beta, _, = self.model_b(s, a_pre, self.a_post_mat)

            q_beta = q_beta.squeeze(2)
            q_beta = q_beta.gather(1, a_index)

            loss_beta = self.loss_beta(beta, a_index.squeeze(1))
            loss_v_beta = r_mc.abs().mean()
            loss_q_beta = self.loss_q_beta(q_beta, r_mc)

            pi, q_pi, _ = self.model(s, a_pre, self.a_post_mat)
            pi_tag, q_tag, _ = self.target(s_tag, a_pre_tag, self.a_post_mat)

            q_pi = q_pi.squeeze(2)
            q_pi = q_pi.gather(1, a_index)

            # ignore negative q:
            if self.double_q:
                _,  q_tag_model, _ = self.model(s_tag, a_pre_tag, self.a_post_mat)
                _, a_tag = F.relu(q_tag_model).max(1)
                q_max = q_tag.gather(1, a_tag.unsqueeze(1))
                q_max = F.relu(q_max).squeeze(1)
            else:
                q_max, _ = F.relu(q_tag).max(1)

            q_max = q_max.detach()

            loss_v_pi = (r + (self.discount ** k) * (q_max * (1 - t))).abs().mean()
            loss_q_pi = self.loss_q_pi(q_pi, r + (self.discount ** k) * (q_max * (1 - t)))
            loss_pi = self.loss_pi(pi, a_index.squeeze(1))

            beta_sfm = F.softmax(beta, 1)
            pi_sfm = F.softmax(pi, 1)
            c = torch.clamp(pi_sfm/beta_sfm, 0, 1)

            c = c.gather(1, a_index)
            weight = (c * q_pi).detach()
            loss_pi = (loss_pi * weight.squeeze(1)).sum()

            # collect actions statistics
            a_index_np = a_index.data.cpu().numpy()

            _, beta_index = beta.data.cpu().max(1)
            beta_index = beta_index.numpy()
            act_diff = (a_index_np != beta_index).astype(np.int)

            R = (r_mc ** 1).mean()

            # add results
            results['act_diff'].append(act_diff)
            results['a_agent'].append(beta_index)
            results['a_player'].append(a_index_np)
            results['loss_beta'].append(loss_beta.data.cpu().numpy()[0])
            results['loss_v_beta'].append((loss_v_beta / R).data.cpu().numpy()[0])
            results['loss_v_pi'].append((loss_v_pi / R).data.cpu().numpy()[0])
            results['loss_q_beta'].append((loss_q_beta / R).data.cpu().numpy()[0])
            # results['loss_q_beta'].append(loss_p.data.cpu().numpy()[0])
            results['loss_pi'].append(loss_pi.data.cpu().numpy()[0])
            results['loss_q_pi'].append((loss_q_pi / R).data.cpu().numpy()[0])
            results['n'].append(n)

            if not (n+1) % n_interval:
                results['s'] = s.data.cpu()
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_agent'] = np.concatenate(results['a_agent'])
                results['a_player'] = np.concatenate(results['a_player'])
                yield results
                self.model.eval()
                self.target.eval()
                self.model_b.eval()
                results = {key: [] for key in results}

    def play_stochastic(self, n_tot):
        raise NotImplementedError

    def play_episode(self, n_tot):

        self.model.eval()
        self.model_b.eval()
        env = Env()

        n_human = 120
        humans_trajectories = iter(self.data)
        softmax = torch.nn.Softmax()

        for i in range(n_tot):

            env.reset()
            observation = next(humans_trajectories)
            trajectory = self.data[observation]
            choices = np.arange(self.global_action_space, dtype=np.int)
            mask = Variable(torch.FloatTensor([0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             requires_grad=False).cuda()
            j = 0
            temp = 1

            while not env.t:

                s = Variable(env.s.cuda(), requires_grad=False)

                beta, vb, qb, _, _ = self.model_b(s, self.actions_matrix)
                pi, v, q, adv, x = self.model(s, self.actions_matrix, beta.detach())

                pi = pi.squeeze(0)
                self.greedy = False

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:
                    eps = np.random.rand()
                    # a = np.random.choice(choices)
                    if self.greedy and eps > 0.1:
                        a = pi.data.cpu().numpy()
                        a = np.argmax(a)
                    else:
                        a = softmax(pi/temp).data.cpu().numpy()
                        a = np.random.choice(choices, p=a)

                q = q[0, a, 0]
                q = q.squeeze(0)

                qb = qb[0, a, 0]
                qb = qb.squeeze(0)

                env.step(a)

                yield {'o': env.s.cpu().numpy(),
                       'v': v.squeeze(0).data.cpu().numpy(),
                       'vb': vb.squeeze(0).data.cpu().numpy(),
                       'qb': qb.squeeze(0).data.cpu().numpy(),
                       's': x[0, :512].data.cpu().numpy(),
                       'score': env.score,
                       'beta': pi.data.cpu().numpy(),
                       'phi': x[0, :512].data.cpu().numpy(),
                       'q': q.squeeze(0).data.cpu().numpy()}

                j += 1

        raise StopIteration

    def policy(self, vs, vl, beta, qs, ql):
        pass
