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
from model import BehavioralDistNet
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationSampler
from agent import Agent
from environment import Env


class BehavioralDistAgent(Agent):

    def __init__(self, load_dataset=True):

        super(BehavioralDistAgent, self).__init__()

        self.meta, self.data = preprocess_demonstrations()

        if load_dataset:
            # demonstration source
            self.meta = divide_dataset(self.meta)

            # datasets
            self.train_dataset = DemonstrationMemory("train", self.meta, self.data)
            self.val_dataset = DemonstrationMemory("val", self.meta, self.data)
            self.test_dataset = DemonstrationMemory("test", self.meta, self.data)
            self.full_dataset = DemonstrationMemory("full", self.meta, self.data)

            self.train_sampler = DemonstrationBatchSampler(self.train_dataset, train=True)
            self.val_sampler = DemonstrationBatchSampler(self.train_dataset, train=False)
            self.test_sampler = DemonstrationBatchSampler(self.test_dataset, train=False)
            self.episodic_sampler = SequentialDemonstrationSampler(self.full_dataset)

            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            num_workers=args.cpu_workers, pin_memory=True, drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_sampler=self.test_sampler,
                                                           num_workers=args.cpu_workers, pin_memory=True, drop_last=False)
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_sampler=self.val_sampler,
                                                          num_workers=args.cpu_workers, pin_memory=True, drop_last=False)

            self.episodic_loader = torch.utils.data.DataLoader(self.full_dataset, sampler=self.episodic_sampler,
                                                               batch_size=self.batch, num_workers=args.cpu_workers)

        if not self.wasserstein:
            self.loss_fn_vs = torch.nn.CrossEntropyLoss(size_average=True)
            self.loss_fn_qs = torch.nn.CrossEntropyLoss(size_average=True)
            self.loss_fn_vl = torch.nn.CrossEntropyLoss(size_average=True)
            self.loss_fn_ql = torch.nn.CrossEntropyLoss(size_average=True)
        else:
            self.loss_fn_vs = wasserstein_metric(support=args.atoms_short, n=1)
            self.loss_fn_qs = wasserstein_metric(support=args.atoms_short, n=1)
            self.loss_fn_vl = wasserstein_metric(support=args.atoms_long, n=1)
            self.loss_fn_ql = wasserstein_metric(support=args.atoms_long, n=1)

        self.histogram = torch.from_numpy(self.meta['histogram']).float()
        m = self.histogram.max()
        self.histogram = m / self.histogram
        self.histogram = torch.clamp(self.histogram, 0, 10).cuda()

        self.loss_fn_beta = torch.nn.CrossEntropyLoss(size_average=True, weight=self.histogram)
        self.loss_fn_pi_s = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)
        self.loss_fn_pi_l = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)
        self.loss_fn_pi_s_tau = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)
        self.loss_fn_pi_l_tau = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)

        # alpha weighted sum

        self.alpha_b = 1  # 1 / 0.7

        self.alpha_vs = 1  # 1 / 0.02
        self.alpha_qs = 1

        self.alpha_vl = 1  # 1 / 0.02
        self.alpha_ql = 1

        self.alpha_pi_s = 1  # 1 / 0.02
        self.alpha_pi_l = 1

        self.alpha_pi_s_tau = 1  # 1 / 0.02
        self.alpha_pi_l_tau = 1

        self.model = BehavioralDistNet()
        self.model.cuda()

        # configure learning

        net_parameters = [p[1] for p in self.model.named_parameters() if "rn_" in p[0]]
        vl_params = [p[1] for p in self.model.named_parameters() if "on_vl" in p[0]]
        ql_params = [p[1] for p in self.model.named_parameters() if "on_ql" in p[0]]
        vs_params = [p[1] for p in self.model.named_parameters() if "on_vs" in p[0]]
        qs_params = [p[1] for p in self.model.named_parameters() if "on_qs" in p[0]]
        beta_params = [p[1] for p in self.model.named_parameters() if "on_beta" in p[0]]

        pi_s_params = [p[1] for p in self.model.named_parameters() if "on_pi_s" in p[0]]
        pi_l_params = [p[1] for p in self.model.named_parameters() if "on_pi_l" in p[0]]
        pi_tau_s_params = [p[1] for p in self.model.named_parameters() if "on_pi_tau_s" in p[0]]
        pi_tau_l_params = [p[1] for p in self.model.named_parameters() if "on_pi_tau_l" in p[0]]

        self.parameters_group_a = net_parameters + vl_params + ql_params + vs_params + qs_params + beta_params
        self.parameters_group_b = pi_s_params + pi_l_params + pi_tau_s_params + pi_tau_l_params

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_vl = BehavioralDistAgent.set_optimizer(net_parameters + vl_params, args.lr_vl)
        self.scheduler_vl = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_vl, self.decay)

        self.optimizer_beta = BehavioralDistAgent.set_optimizer(net_parameters + beta_params, args.lr_beta)
        self.scheduler_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta, self.decay)

        self.optimizer_vs = BehavioralDistAgent.set_optimizer(net_parameters + vs_params, args.lr_vs)
        self.scheduler_vs = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_vs, self.decay)

        self.optimizer_qs = BehavioralDistAgent.set_optimizer(net_parameters + qs_params, args.lr_qs)
        self.scheduler_qs = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_qs, self.decay)

        self.optimizer_ql = BehavioralDistAgent.set_optimizer(net_parameters + ql_params, args.lr_ql)
        self.scheduler_ql = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_ql, self.decay)

        self.optimizer_pi_l = BehavioralDistAgent.set_optimizer(pi_l_params, args.lr_pi_l)
        self.scheduler_pi_l = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi_l, self.decay)

        self.optimizer_pi_s = BehavioralDistAgent.set_optimizer(pi_s_params, args.lr_pi_s)
        self.scheduler_pi_s = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi_s, self.decay)

        self.optimizer_pi_l_tau = BehavioralDistAgent.set_optimizer(pi_tau_l_params, args.lr_pi_tau_l)
        self.scheduler_pi_l_tau = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi_l_tau, self.decay)

        self.optimizer_pi_s_tau = BehavioralDistAgent.set_optimizer(pi_tau_s_params, args.lr_pi_tau_s)
        self.scheduler_pi_s_tau = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi_s_tau, self.decay)

        actions = torch.FloatTensor(consts.hotvec_matrix) / (3**(0.5))
        actions = Variable(actions, requires_grad=False).cuda()

        self.actions_matrix = actions.unsqueeze(0)
        self.reverse_excitation_index = consts.hotvec_inv

        self.short_bins = consts.short_bins[args.game][:-1] / self.meta['avg_score']
        # the long bins are already normalized
        self.long_bins = consts.long_bins[args.game][:-1]

        self.short_bins_torch = Variable(torch.from_numpy(consts.short_bins[args.game] / self.meta['avg_score']), requires_grad=False).cuda()
        self.long_bins_torch = Variable(torch.from_numpy(consts.long_bins[args.game]), requires_grad=False).cuda()

        self.batch_range = np.arange(self.batch)

        self.zero = Variable(torch.zeros(1))

    def flip_grad(self, parameters):
        for p in parameters:
            p.requires_grad = not p.requires_grad

    @staticmethod
    def individual_loss_fn_l2(argument):
        return abs(argument.data.cpu().numpy())**2

    @staticmethod
    def individual_loss_fn_l1(argument):
        return abs(argument.data.cpu().numpy())

    def save_checkpoint(self, path, aux=None):

        cpu_state = self.model.state_dict()
        for k in cpu_state:
            cpu_state[k] = cpu_state[k].cpu()

        state = {'state_dict': self.model.state_dict(),
                 'state_dict_cpu': cpu_state,
                 'optimizer_vl_dict': self.optimizer_vl.state_dict(),
                 'optimizer_beta_dict': self.optimizer_beta.state_dict(),
                 'optimizer_vs_dict': self.optimizer_vs.state_dict(),
                 'optimizer_ql_dict': self.optimizer_ql.state_dict(),
                 'optimizer_qs_dict': self.optimizer_qs.state_dict(),
                 'optimizer_pi_s_dict': self.optimizer_pi_s.state_dict(),
                 'optimizer_pi_l_dict': self.optimizer_pi_l.state_dict(),
                 'optimizer_pi_s_tau_dict': self.optimizer_pi_s_tau.state_dict(),
                 'optimizer_pi_l_tau_dict': self.optimizer_pi_l_tau.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def one_hot(self, y, nb_digits):
        batch_size = y.shape[0]
        y_onehot = torch.zeros(batch_size, nb_digits)
        return y_onehot.scatter_(1, y.unsqueeze(1), 1)

    def load_checkpoint(self, path):

        if self.cuda:
            state = torch.load(path)
            self.model.load_state_dict(state['state_dict'])
        else:
            state = torch.load(path, map_location=lambda storage, location: storage)
            self.model.load_state_dict(state['state_dict_cpu'])
        self.optimizer_vl.load_state_dict(state['optimizer_vl_dict'])
        self.optimizer_beta.load_state_dict(state['optimizer_beta_dict'])
        self.optimizer_vs.load_state_dict(state['optimizer_vs_dict'])
        self.optimizer_ql.load_state_dict(state['optimizer_ql_dict'])
        self.optimizer_qs.load_state_dict(state['optimizer_qs_dict'])
        self.optimizer_pi_s_tau.load_state_dict(state['optimizer_pi_s_tau_dict'])
        self.optimizer_pi_l_tau.load_state_dict(state['optimizer_pi_l_tau_dict'])
        self.optimizer_pi_s.load_state_dict(state['optimizer_pi_s_dict'])
        self.optimizer_pi_l.load_state_dict(state['optimizer_pi_l_dict'])

        return state['aux']

    def resume(self, model_path):

        aux = self.load_checkpoint(model_path)
        # self.update_target()
        return aux

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def dummy_episodic_evaluator(self):
        while True:
            yield {'q_diff': torch.zeros(100), 'a_agent': torch.zeros(100, self.action_space), 'a_player': torch.zeros(100).long()}

    def _episodic_evaluator(self):
        pass

    def get_weighted_loss(self, x, bins):
        xd = x.data
        inds = xd.cumsum(1) <= self.quantile
        inds = inds.sum(1).long()
        return bins[inds]

    def learn(self, n_interval, n_tot):

        self.model.train()
        # self.target.eval()
        results = {'n': [], 'loss_vs': [], 'loss_b': [], 'loss_vl': [],
                   'loss_qs': [], 'loss_ql': [], 'loss_pi_s': [], 'loss_pi_l': [],
                   'loss_pi_s_tau': [], 'loss_pi_l_tau': []}

        self.flip_grad(self.parameters_group_b)
        train_net = True

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda(), requires_grad=False)

            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)

            rl = np.digitize(sample['score'].numpy(), self.long_bins, right=True)
            rs = np.digitize(sample['f'].numpy(), self.short_bins, right=True)

            Rl = Variable(sample['score'].cuda(), requires_grad=False)
            Rs = Variable(sample['f'].cuda(), requires_grad=False)

            if self.wasserstein:
                rl = Variable(self.one_hot(torch.LongTensor(rl), self.atoms_long).cuda(), requires_grad=False)
                rs = Variable(self.one_hot(torch.LongTensor(rs), self.atoms_short).cuda(), requires_grad=False)
            else:
                rl = Variable(torch.from_numpy(rl).cuda(), requires_grad=False)
                rs = Variable(torch.from_numpy(rs).cuda(), requires_grad=False)

            vs, vl, beta, qs, ql, phi, pi_s, pi_l, pi_s_tau, pi_l_tau = self.model(s, a)

            # policy learning

            if self.alpha_vs and train_net:
                loss_vs = self.alpha_vs * self.loss_fn_vs(vs, rs)
                self.optimizer_vs.zero_grad()
                loss_vs.backward(retain_graph=True)
                self.optimizer_vs.step()
            else:
                loss_vs = self.zero

            if self.alpha_vl and train_net:
                loss_vl = self.alpha_vl * self.loss_fn_vl(vl, rl)
                self.optimizer_vl.zero_grad()
                loss_vl.backward(retain_graph=True)
                self.optimizer_vl.step()
            else:
                loss_vl = self.zero

            if self.alpha_b and train_net:
                loss_b = self.alpha_b * self.loss_fn_beta(beta, a_index)
                self.optimizer_beta.zero_grad()
                loss_b.backward(retain_graph=True)
                self.optimizer_beta.step()
            else:
                loss_b = self.zero

            if self.alpha_qs and train_net:
                loss_qs = self.alpha_qs * self.loss_fn_qs(qs, rs)
                self.optimizer_qs.zero_grad()
                loss_qs.backward(retain_graph=True)
                self.optimizer_qs.step()
            else:
                loss_qs = self.zero

            if self.alpha_ql and train_net:
                loss_ql = self.alpha_ql * self.loss_fn_ql(ql, rl)
                self.optimizer_ql.zero_grad()
                loss_ql.backward(retain_graph=True)
                self.optimizer_ql.step()
            else:
                loss_ql = self.zero

            a_index_np = sample['a_index'].numpy()
            self.batch_range = np.arange(self.batch)

            beta_sfm = F.softmax(beta, 1)
            pi_s_sfm = F.softmax(pi_s, 1)
            pi_l_sfm = F.softmax(pi_l, 1)
            pi_s_tau_sfm = F.softmax(pi_s, 1)
            pi_l_tau_sfm = F.softmax(pi_l, 1)

            beta_fix = Variable(beta_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_s_fix = Variable(pi_s_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_l_fix = Variable(pi_l_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_s_tau_fix = Variable(pi_s_tau_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_l_tau_fix = Variable(pi_l_tau_sfm.data[self.batch_range, a_index_np], requires_grad=False)

            if self.alpha_pi_s and not train_net:
                loss_pi_s = self.alpha_pi_s * self.loss_fn_pi_s(pi_s, a_index)
                loss_pi_s = (loss_pi_s * Rs * self.off_factor(pi_s_fix, beta_fix)).mean()
                self.optimizer_pi_s.zero_grad()
                loss_pi_s.backward(retain_graph=True)
                self.optimizer_pi_s.step()
            else:
                loss_pi_s = self.zero

            if self.alpha_pi_l and not train_net:
                loss_pi_l = self.alpha_pi_l * self.loss_fn_pi_l(pi_l, a_index)
                loss_pi_l = (loss_pi_l * Rl * self.off_factor(pi_l_fix, beta_fix)).mean()
                self.optimizer_pi_l.zero_grad()
                loss_pi_l.backward(retain_graph=True)
                self.optimizer_pi_l.step()
            else:
                loss_pi_l = self.zero

            if self.alpha_pi_s_tau and not train_net:
                loss_pi_s_tau = self.alpha_pi_s_tau * self.loss_fn_pi_s_tau(pi_s_tau, a_index)
                w = self.get_weighted_loss(F.softmax(qs, 1), self.short_bins_torch)
                loss_pi_s_tau = (loss_pi_s_tau * w * self.off_factor(pi_s_tau_fix, beta_fix)).mean()
                self.optimizer_pi_s_tau.zero_grad()
                loss_pi_s_tau.backward(retain_graph=True)
                self.optimizer_pi_s_tau.step()
            else:
                loss_pi_s_tau = self.zero

            if self.alpha_pi_l_tau and not train_net:
                loss_pi_l_tau = self.alpha_pi_l_tau * self.loss_fn_pi_l_tau(pi_l_tau, a_index)
                w = self.get_weighted_loss(F.softmax(ql, 1), self.long_bins_torch)
                loss_pi_l_tau = (loss_pi_l_tau * w * self.off_factor(pi_l_tau_fix, beta_fix)).mean()
                self.optimizer_pi_l_tau.zero_grad()
                loss_pi_l_tau.backward()
                self.optimizer_pi_l_tau.step()
            else:
                loss_pi_l_tau = self.zero

            # add results
            results['loss_vs'].append(loss_vs.data.cpu().numpy()[0])
            results['loss_vl'].append(loss_vl.data.cpu().numpy()[0])
            results['loss_b'].append(loss_b.data.cpu().numpy()[0])
            results['loss_qs'].append(loss_qs.data.cpu().numpy()[0])
            results['loss_ql'].append(loss_ql.data.cpu().numpy()[0])
            results['loss_pi_s'].append(loss_pi_s.data.cpu().numpy()[0])
            results['loss_pi_l'].append(loss_pi_l.data.cpu().numpy()[0])
            results['loss_pi_s_tau'].append(loss_pi_s_tau.data.cpu().numpy()[0])
            results['loss_pi_l_tau'].append(loss_pi_l_tau.data.cpu().numpy()[0])
            results['n'].append(n)

            # if not n % self.update_target_interval:
            #     # self.update_target()

            # if an index is rolled more than once during update_memory_interval period, only the last occurance affect the
            if not (n+1) % self.update_memory_interval and self.prioritized_replay:
                self.train_dataset.update_probabilities()

            # update a global n_step parameter

            if not (n+1) % self.update_n_steps_interval:
                # self.train_dataset.update_n_step(n + 1)
                d = np.divmod(n+1, self.update_n_steps_interval)[0]
                if d % 10 == 1:
                    self.flip_grad(self.parameters_group_b + self.parameters_group_a)
                    train_net = not train_net
                if d % 10 == 2:
                    self.flip_grad(self.parameters_group_b + self.parameters_group_a)
                    train_net = not train_net

                    self.scheduler_pi_s.step()
                    self.scheduler_pi_l.step()
                    self.scheduler_pi_s_tau.step()
                    self.scheduler_pi_l_tau.step()
                else:
                    self.scheduler_vs.step()
                    self.scheduler_beta.step()
                    self.scheduler_vl.step()
                    self.scheduler_qs.step()
                    self.scheduler_ql.step()


            if not (n+1) % n_interval:
                yield results
                self.model.train()
                # self.target.eval()
                results = {key: [] for key in results}

    def off_factor(self, pi, beta):
        return torch.clamp(pi/beta, 0, 1)


    def test(self, n_interval, n_tot):

        self.model.eval()
        # self.target.eval()

        results = {'n': [], 'loss_vs': [], 'loss_b': [], 'loss_vl': [], 'loss_qs': [],
                   'loss_ql': [], 'act_diff': [], 'a_agent': [], 'a_player': [],
                   'loss_pi_s': [], 'loss_pi_l': [], 'loss_pi_s_tau': [], 'loss_pi_l_tau': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)

            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)

            rl = np.digitize(sample['score'].numpy(), self.long_bins, right=True)
            rs = np.digitize(sample['f'].numpy(), self.short_bins, right=True)

            Rl = Variable(sample['score'].cuda(), requires_grad=False)
            Rs = Variable(sample['f'].cuda(), requires_grad=False)

            if self.wasserstein:
                rl = Variable(self.one_hot(torch.LongTensor(rl), self.atoms_long).cuda(), requires_grad=False)
                rs = Variable(self.one_hot(torch.LongTensor(rs), self.atoms_short).cuda(), requires_grad=False)
            else:
                rl = Variable(torch.from_numpy(rl).cuda(), requires_grad=False)
                rs = Variable(torch.from_numpy(rs).cuda(), requires_grad=False)

            vs, vl, beta, qs, ql, phi, pi_s, pi_l, pi_s_tau, pi_l_tau = self.model(s, a)

            qs = qs.squeeze(1)
            ql = ql.squeeze(1)

            # policy learning

            loss_vs = self.alpha_vs * self.loss_fn_vs(vs, rs)
            loss_vl = self.alpha_vl * self.loss_fn_vl(vl, rl)
            loss_b = self.alpha_b * self.loss_fn_beta(beta, a_index)
            loss_qs = self.alpha_qs * self.loss_fn_qs(qs, rs)
            loss_ql = self.alpha_ql * self.loss_fn_ql(ql, rl)

            a_index_np = sample['a_index'].numpy()
            self.batch_range = np.arange(self.batch)

            beta_sfm = F.softmax(beta, 1)
            pi_s_sfm = F.softmax(pi_s, 1)
            pi_l_sfm = F.softmax(pi_l, 1)
            pi_s_tau_sfm = F.softmax(pi_s, 1)
            pi_l_tau_sfm = F.softmax(pi_l, 1)

            beta_fix = Variable(beta_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_s_fix = Variable(pi_s_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_l_fix = Variable(pi_l_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_s_tau_fix = Variable(pi_s_tau_sfm.data[self.batch_range, a_index_np], requires_grad=False)
            pi_l_tau_fix = Variable(pi_l_tau_sfm.data[self.batch_range, a_index_np], requires_grad=False)

            loss_pi_s = self.alpha_pi_s * self.loss_fn_pi_s(pi_s, a_index)
            loss_pi_s = (loss_pi_s * Rs * self.off_factor(pi_s_fix ,beta_fix)).mean()

            loss_pi_l = self.alpha_pi_l * self.loss_fn_pi_l(pi_l, a_index)
            loss_pi_l = (loss_pi_l * Rl * self.off_factor(pi_l_fix ,beta_fix)).mean()

            loss_pi_s_tau = self.alpha_pi_s_tau * self.loss_fn_pi_s_tau(pi_s_tau, a_index)
            w = self.get_weighted_loss(F.softmax(qs, 1), self.short_bins_torch)
            loss_pi_s_tau = (loss_pi_s_tau * w * self.off_factor(pi_s_tau_fix ,beta_fix)).mean()

            loss_pi_l_tau = self.alpha_pi_l_tau * self.loss_fn_pi_l_tau(pi_l_tau, a_index)
            w = self.get_weighted_loss(F.softmax(ql, 1), self.long_bins_torch)
            loss_pi_l_tau = (loss_pi_l_tau * w * self.off_factor(pi_l_tau_fix ,beta_fix)).mean()

            # collect actions statistics
            a_index_np = a_index.data.cpu().numpy()

            _, beta_index = beta.data.cpu().max(1)
            beta_index = beta_index.numpy()
            act_diff = (a_index_np != beta_index).astype(np.int)

            # add results
            results['act_diff'].append(act_diff)
            results['a_agent'].append(beta_index)
            results['a_player'].append(a_index_np)
            results['loss_vs'].append(loss_vs.data.cpu().numpy()[0])
            results['loss_vl'].append(loss_vl.data.cpu().numpy()[0])
            results['loss_b'].append(loss_b.data.cpu().numpy()[0])
            results['loss_qs'].append(loss_qs.data.cpu().numpy()[0])
            results['loss_ql'].append(loss_ql.data.cpu().numpy()[0])
            results['loss_pi_s'].append(loss_pi_s.data.cpu().numpy()[0])
            results['loss_pi_l'].append(loss_pi_l.data.cpu().numpy()[0])
            results['loss_pi_s_tau'].append(loss_pi_s_tau.data.cpu().numpy()[0])
            results['loss_pi_l_tau'].append(loss_pi_l_tau.data.cpu().numpy()[0])
            results['n'].append(n)

            if not (n+1) % n_interval:
                results['s'] = s.data.cpu()
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_agent'] = np.concatenate(results['a_agent'])
                results['a_player'] = np.concatenate(results['a_player'])
                yield results
                self.model.eval()
                # self.target.eval()
                results = {key: [] for key in results}

    def play_stochastic(self, n_tot):
        raise NotImplementedError
        # self.model.eval()
        # env = Env()
        # render = args.render
        #
        # n_human = 60
        # humans_trajectories = iter(self.data)
        #
        # for i in range(n_tot):
        #
        #     env.reset()
        #
        #     observation = next(humans_trajectories)
        #     print("Observation %s" % observation)
        #     trajectory = self.data[observation]
        #
        #     j = 0
        #
        #     while not env.t:
        #
        #         if j < n_human:
        #             a = trajectory[j, self.meta['action']]
        #
        #         else:
        #
        #             if self.cuda:
        #                 s = Variable(env.s.cuda(), requires_grad=False)
        #             else:
        #                 s = Variable(env.s, requires_grad=False)
        #             _, q, _, _, _, _ = self.model(s, self.actions_matrix)
        #
        #             q = q.squeeze(2)
        #
        #             q = q.data.cpu().numpy()
        #             a = np.argmax(q)
        #
        #         env.step(a)
        #
        #         j += 1
        #
        #     yield {'o': env.s.cpu().numpy(),
        #            'score': env.score}

    def play_episode(self, n_tot):

        self.model.eval()
        env = Env()

        n_human = 120
        humans_trajectories = iter(self.data)
        softmax = torch.nn.Softmax()

        # mask = torch.FloatTensor(consts.actions_mask[args.game])
        # mask = Variable(mask.cuda(), requires_grad=False)

        vsx = torch.FloatTensor(consts.short_bins[args.game])
        vlx = torch.FloatTensor(consts.long_bins[args.game])

        for i in range(n_tot):

            env.reset()
            observation = next(humans_trajectories)
            trajectory = self.data[observation]
            choices = np.arange(self.global_action_space, dtype=np.int)

            j = 0

            while not env.t:

                s = Variable(env.s.cuda(), requires_grad=False)
                vs, vl, beta, qs, ql, phi, pi_s, pi_l, pi_s_tau, pi_l_tau = self.model(s, self.actions_matrix)
                beta = beta.squeeze(0)
                pi_l = pi_l.squeeze(0)
                pi_s = pi_s.squeeze(0)
                pi_l_tau = pi_l_tau.squeeze(0)
                pi_s_tau = pi_s_tau.squeeze(0)


                temp = 1

                # consider only 3 most frequent actions
                beta_np = beta.data.cpu().numpy()
                indices = np.argsort(beta_np)

                maskb = Variable(torch.FloatTensor([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), requires_grad=False).cuda()
                # maskb = Variable(torch.FloatTensor([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                #                  requires_grad=False).cuda()

                # pi = maskb * (beta / beta.max())

                pi = beta
                self.greedy = False

                beta_prob = pi

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

                env.step(a)

                vs = softmax(vs)
                vl = softmax(vl)
                vs = torch.sum(vsx * vs.data.cpu())
                vl = torch.sum(vlx * vl.data.cpu())

                yield {'o': env.s.cpu().numpy(),
                       'vs': np.array([vs]),
                       'vl': np.array([vl]),
                       's': phi.data.cpu().numpy(),
                       'score': env.score,
                       'beta': beta_prob.data.cpu().numpy(),
                       'phi': phi.squeeze(0).data.cpu().numpy(),
                       'qs':  qs.squeeze(0).data.cpu().numpy(),
                       'ql': ql.squeeze(0).data.cpu().numpy(),}

                j += 1

        raise StopIteration

    def policy(self, vs, vl, beta, qs, ql):
        pass
