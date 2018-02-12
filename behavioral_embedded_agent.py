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
from model import ActorCritic, BehavioralDistEmbedding
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationSampler
from agent import Agent
from environment import Env
from preprocess import calc_hist_weights


class BehavioralEmbeddedAgent(Agent):

    def __init__(self, load_dataset=True):

        super(BehavioralEmbeddedAgent, self).__init__()

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

        self.loss_v_beta = torch.nn.KLDivLoss()
        self.loss_q_beta = torch.nn.KLDivLoss()

        self.loss_v_pi = torch.nn.KLDivLoss()
        self.loss_q_pi = torch.nn.KLDivLoss()

        self.histogram = torch.from_numpy(self.meta['histogram']).float()

        w_f, w_v, w_h = calc_hist_weights(self.histogram)

        w_f = torch.clamp(w_f, 0, 10).cuda()
        w_v = torch.clamp(w_v, 0, 10).cuda()
        w_h = torch.clamp(w_h, 0, 10).cuda()

        self.loss_beta_f = torch.nn.CrossEntropyLoss(size_average=True, weight=w_f)
        self.loss_beta_v = torch.nn.CrossEntropyLoss(size_average=True, weight=w_v)
        self.loss_beta_h = torch.nn.CrossEntropyLoss(size_average=True, weight=w_h)

        self.loss_pi_f = torch.nn.CrossEntropyLoss(size_average=False)
        self.loss_pi_v = torch.nn.CrossEntropyLoss(size_average=False)
        self.loss_pi_h = torch.nn.CrossEntropyLoss(size_average=False)



        self.behavioral_model = BehavioralDistEmbedding()
        self.behavioral_model.cuda()

        # actor critic setting

        self.actor_critic_model = ActorCritic()
        self.actor_critic_model.cuda()

        self.actor_critic_target = ActorCritic()
        self.actor_critic_target.cuda()

        # configure learning

        cnn_params = [p[1] for p in self.behavioral_model.named_parameters() if "cnn" in p[0]]
        emb_params = [p[1] for p in self.behavioral_model.named_parameters() if "emb" in p[0]]

        v_beta_params = [p[1] for p in self.behavioral_model.named_parameters() if "fc_v" in p[0]]
        a_beta_params = [p[1] for p in self.behavioral_model.named_parameters() if "fc_adv" in p[0]]

        beta_f_params = [p[1] for p in self.behavioral_model.named_parameters() if "fc_beta_f" in p[0]]
        beta_v_params = [p[1] for p in self.behavioral_model.named_parameters() if "fc_beta_v" in p[0]]
        beta_h_params = [p[1] for p in self.behavioral_model.named_parameters() if "fc_beta_h" in p[0]]

        v_pi_params = [p[1] for p in self.actor_critic_model.named_parameters() if "critic_v" in p[0]]
        a_pi_params = [p[1] for p in self.actor_critic_model.named_parameters() if "critic_adv" in p[0]]

        pi_f_params = [p[1] for p in self.actor_critic_model.named_parameters() if "fc_actor_f" in p[0]]
        pi_v_params = [p[1] for p in self.actor_critic_model.named_parameters() if "fc_actor_v" in p[0]]
        pi_h_params = [p[1] for p in self.actor_critic_model.named_parameters() if "fc_actor_h" in p[0]]

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER

        self.optimizer_critic_v = BehavioralEmbeddedAgent.set_optimizer(v_pi_params, 0.0008)
        self.scheduler_critic_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_critic_v, self.decay)

        self.optimizer_critic_q = BehavioralEmbeddedAgent.set_optimizer(v_pi_params + a_pi_params, 0.0008)
        self.scheduler_critic_q = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_critic_q, self.decay)

        self.optimizer_v_beta = BehavioralEmbeddedAgent.set_optimizer(cnn_params + emb_params + v_beta_params, 0.0008)
        self.scheduler_v_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v_beta, self.decay)

        self.optimizer_q_beta = BehavioralEmbeddedAgent.set_optimizer(cnn_params + emb_params + v_beta_params + a_beta_params, 0.0008)
        self.scheduler_q_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q_beta, self.decay)

        self.optimizer_beta_f = BehavioralEmbeddedAgent.set_optimizer(cnn_params + emb_params + beta_f_params, 0.0008)
        self.scheduler_beta_f = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta_f, self.decay)

        self.optimizer_beta_v = BehavioralEmbeddedAgent.set_optimizer(cnn_params + emb_params + beta_v_params, 0.0008)
        self.scheduler_beta_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta_v, self.decay)

        self.optimizer_beta_h = BehavioralEmbeddedAgent.set_optimizer(cnn_params + emb_params + beta_h_params, 0.0008)
        self.scheduler_beta_h = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta_h, self.decay)

        self.optimizer_pi_f = BehavioralEmbeddedAgent.set_optimizer(pi_f_params, 0.0008)
        self.scheduler_pi_f = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi_f, self.decay)

        self.optimizer_pi_v = BehavioralEmbeddedAgent.set_optimizer(pi_v_params, 0.0008)
        self.scheduler_pi_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi_v, self.decay)

        self.optimizer_pi_h = BehavioralEmbeddedAgent.set_optimizer(pi_h_params, 0.0008)
        self.scheduler_pi_h = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pi_h, self.decay)

        actions = torch.LongTensor(consts.hotvec_matrix).cuda()
        self.actions_matrix = actions.unsqueeze(0)

        self.q_bins = consts.q_bins[args.game][:-1] / self.meta['avg_score']
        # the long bins are already normalized
        self.v_bins = consts.v_bins[args.game][:-1] / self.meta['avg_score']

        self.q_bins_torch = Variable(torch.from_numpy(consts.q_bins[args.game] / self.meta['avg_score']), requires_grad=False).cuda()
        self.v_bins_torch = Variable(torch.from_numpy(consts.v_bins[args.game] / self.meta['avg_score']), requires_grad=False).cuda()

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

        state = {'behavioral_model': self.behavioral_model.state_dict(),
                 'actor_critic_model': self.actor_critic_model.state_dict(),
                 'optimizer_critic_v': self.optimizer_critic_v.state_dict(),
                 'optimizer_critic_q': self.optimizer_critic_q.state_dict(),
                 'optimizer_v_beta': self.optimizer_v_beta.state_dict(),
                 'optimizer_q_beta': self.optimizer_q_beta.state_dict(),
                 'optimizer_beta_f': self.optimizer_beta_f.state_dict(),
                 'optimizer_beta_v': self.optimizer_beta_v.state_dict(),
                 'optimizer_beta_h': self.optimizer_beta_h.state_dict(),
                 'optimizer_pi_f': self.optimizer_pi_f.state_dict(),
                 'optimizer_pi_v': self.optimizer_pi_v.state_dict(),
                 'optimizer_pi_h': self.optimizer_pi_h.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path)
        self.behavioral_model.load_state_dict(state['behavioral_model'])
        self.actor_critic_model.load_state_dict(state['actor_critic_model'])
        self.optimizer_critic_v.load_state_dict(state['optimizer_critic_v'])
        self.optimizer_critic_q.load_state_dict(state['optimizer_critic_q'])
        self.optimizer_v_beta.load_state_dict(state['optimizer_v_beta'])
        self.optimizer_q_beta.load_state_dict(state['optimizer_q_beta'])
        self.optimizer_beta_f.load_state_dict(state['optimizer_beta_f'])
        self.optimizer_beta_v.load_state_dict(state['optimizer_beta_v'])
        self.optimizer_beta_h.load_state_dict(state['optimizer_beta_h'])
        self.optimizer_pi_f.load_state_dict(state['optimizer_pi_f'])
        self.optimizer_pi_v.load_state_dict(state['optimizer_pi_v'])
        self.optimizer_pi_h.load_state_dict(state['optimizer_pi_h'])

        return state['aux']

    def resume(self, model_path):

        aux = self.load_checkpoint(model_path)
        # self.update_target()
        return aux

    def update_target(self):
        self.actor_critic_target.load_state_dict(self.actor_critic_model.state_dict())

    def batched_interp(self, x, xp, fp):
        # implemented with numpy
        x = x.data.cpu().numpy()
        xp = xp.data.cpu().numpy()
        fp = fp.data.cpu().numpy()
        y = np.zeros(x.shape)

        for i, (xl, xpl, fpl) in enumerate(zip(x, xp, fp)):
            y[i] = np.interp(xl, xpl, fpl)

        return Variable(torch.FloatTensor().cuda(), requires_grad=False)

    def new_distribution(self, q, beta, r, bin):
        bin = bin.repeat(self.batch, self.global_action_space, 1)
        r = r.unsqueeze(1).repeat(1, bin.shape[0])
        beta = beta.unsqueeze(1)

        # dimensions:
        # bins [batch, actions, bins]
        # beta [batch, 1, actions]
        # new_bin = torch.baddbmm(r, beta, , alpha=self.discount)
        q_back.squeeze(1)
        return self.batched_interp(x, xp, fp)


    def learn(self, n_interval, n_tot):

        self.behavioral_model.train()
        self.actor_critic_model.train()
        self.actor_critic_target.eval()

        results = {'n': [], 'loss_v': [], 'loss_q': [], 'loss_beta_f': [],
                   'loss_beta_v': [], 'loss_beta_h': [], 'loss_pi_s': [], 'loss_pi_l': [],
                   'loss_pi_s_tau': [], 'loss_pi_l_tau': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda(), requires_grad=False)

            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)

            rl = np.digitize(sample['score'].numpy(), self.long_bins, right=True)
            rs = np.digitize(sample['f'].numpy(), self.short_bins, right=True)

            Rl = Variable(sample['score'].cuda(), requires_grad=False)
            Rs = Variable(sample['f'].cuda(), requires_grad=False)

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

                maskb = Variable(torch.FloatTensor([0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), requires_grad=False).cuda()
                # maskb = Variable(torch.FloatTensor([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                #                  requires_grad=False).cuda()

                # pi = maskb * (beta / beta.max())

                pi = beta
                self.greedy = True

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
