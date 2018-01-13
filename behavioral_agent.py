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
from model import BehavioralNet, BehavioralNetDeterministic
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationSampler
from player import player_worker, QPlayer, AVPlayer, AVAPlayer
from agent import Agent
from environment import Env

class BehavioralAgent(Agent):

    def __init__(self, load_dataset=True):

        super(BehavioralAgent, self).__init__()

        self.actions_transform = np.array(consts.action2activation[args.game])

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

        if self.l1_loss:
            self.loss_fn_value = torch.nn.L1Loss(size_average=True)
            self.individual_loss_fn_value = self.individual_loss_fn_l1
        else:
            self.loss_fn_value = torch.nn.MSELoss(size_average=True)
            self.individual_loss_fn_value = self.individual_loss_fn_l2

        self.loss_fn_r = torch.nn.MSELoss(size_average=True)
        self.individual_loss_fn_r = self.individual_loss_fn_l2

        self.loss_fn_q = torch.nn.L1Loss(size_average=True)
        self.individual_loss_fn_q = self.individual_loss_fn_l1

        self.loss_fn_p = torch.nn.L1Loss(size_average=True)
        self.individual_loss_fn_p = self.individual_loss_fn_l1

        # self.target_single = BehavioralNet(self.global_action_space)

        # alpha weighted sum

        self.alpha_v = 1  # 1 / 0.02
        self.alpha_b = 1  # 1 / 0.7

        self.alpha_r = 1  # 1 / 0.7
        self.alpha_p = 1  # 1 / 0.7
        self.alpha_q = 1

        if args.deterministic:  # 1 / 0.02
            self.loss_fn_beta = torch.nn.L1Loss(size_average=True)
            self.learn = self.learn_deterministic
            self.test = self.test_deterministic
            self.play = self.play_deterministic
            self.play_episode = self.play_episode_deterministic
            self.model_single = BehavioralNetDeterministic(self.global_action_space)

        else:
            self.loss_fn_beta = torch.nn.CrossEntropyLoss()
            self.learn = self.learn_stochastic
            self.test = self.test_stochastic
            self.play = self.play_stochastic
            self.play_episode = self.play_episode_stochastic
            self.model_single = BehavioralNet(self.global_action_space)

        # configure learning

        if self.cuda:
            self.model_single = self.model_single.cuda()
            # self.model = torch.nn.DataParallel(self.model_single)
            self.model = self.model_single
            # self.target_single = self.target_single.cuda()
            # self.target = torch.nn.DataParallel(self.target_single)
        else:
            self.model = self.model_single
            # self.target = self.target_single

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_v = BehavioralAgent.set_optimizer(self.model.parameters(), args.lr)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v, self.decay)

        self.optimizer_beta = BehavioralAgent.set_optimizer(self.model.parameters(), args.lr_beta)
        self.scheduler_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta, self.decay)

        self.optimizer_q = BehavioralAgent.set_optimizer(self.model.parameters(), args.lr_q)
        self.scheduler_q = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q, self.decay)

        self.optimizer_r = BehavioralAgent.set_optimizer(self.model.parameters(), args.lr_r)
        self.scheduler_r = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_r, self.decay)

        self.optimizer_p = BehavioralAgent.set_optimizer(self.model.parameters(), args.lr_p)
        self.scheduler_p = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_p, self.decay)

        self.episodic_evaluator = self.dummy_episodic_evaluator

        # build the action matrix
        # excitation = torch.LongTensor(consts.game_excitation_map[args.game])
        excitation = torch.LongTensor(consts.excitation_map)
        mask = torch.LongTensor(consts.excitation_mask[args.game])
        mask_dup = mask.unsqueeze(0).repeat(consts.action_space, 1)
        actions = Variable(mask_dup * excitation, requires_grad=False)
        actions = Variable(excitation, requires_grad=False)
        if args.cuda:
            actions = actions.cuda()

        self.actions_matrix = actions.unsqueeze(0)
        self.actions_matrix = self.actions_matrix.repeat(1, 1, 1).float()

        self.go_to_max = np.inf # 4096 * 8 * 4

        self.reverse_excitation_index = consts.reverse_excitation_index

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
                 'optimizer_v_dict': self.optimizer_v.state_dict(),
                 'optimizer_beta_dict': self.optimizer_beta.state_dict(),
                 'optimizer_p_dict': self.optimizer_p.state_dict(),
                 'optimizer_qeta_dict': self.optimizer_q.state_dict(),
                 'optimizer_r_dict': self.optimizer_r.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        if self.cuda:
            state = torch.load(path)
            self.model.load_state_dict(state['state_dict'])
        else:
            state = torch.load(path, map_location=lambda storage, location: storage)
            self.model.load_state_dict(state['state_dict_cpu'])
        self.optimizer_v.load_state_dict(state['optimizer_v_dict'])
        self.optimizer_beta.load_state_dict(state['optimizer_beta_dict'])
        self.optimizer_p.load_state_dict(state['optimizer_p_dict'])
        self.optimizer_q.load_state_dict(state['optimizer_qeta_dict'])
        self.optimizer_r.load_state_dict(state['optimizer_r_dict'])

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

    def learn_stochastic(self, n_interval, n_tot):

        self.model.train()
        # self.target.eval()
        results = {'n': [], 'loss_v': [], 'loss_b': [], 'loss_q': [], 'loss_p': [], 'loss_r': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].float().cuda(), requires_grad=False)
            a_tag = Variable(sample['a_tag'].float().cuda(), requires_grad=False)
            r = Variable(sample['r'].float().cuda().unsqueeze(1), requires_grad=False)
            t = Variable(sample['t'].float().cuda().unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].float().cuda().unsqueeze(1), requires_grad=False)
            a_index = Variable(sample['a_index'].cuda(), requires_grad=False)
            f = Variable(sample['f'].float().cuda().unsqueeze(1), requires_grad=False)
            indexes = sample['i']

            value, q, beta, reward, p, phi = self.model(s, a)

            _, _, _, _, _, phi_tag = self.model(s_tag, a_tag)

            # m = (((f - value) > 0).float() + 1) if n > self.go_to_max else Variable(torch.ones(f.data.shape).cuda())
            # m = m.detach()

            loss_v = self.alpha_v * self.loss_fn_value(value, f)
            loss_q = self.alpha_q * self.loss_fn_q(q, f)

            loss_b = self.alpha_b * self.loss_fn_beta(beta, a_index)

            loss_r = self.alpha_r * self.loss_fn_r(reward, r)

            phi_tag = phi_tag.detach()
            loss_p = self.alpha_p * self.loss_fn_p(p, phi_tag)

            #
            # # calculate the td error for the priority replay
            # if self.prioritized_replay:
            #     argument = r + (self.discount ** k) * (max_q_target * (1 - t)) - q_a
            #     individual_loss = LfdAgent.individual_loss_fn(argument)
            #     self.train_dataset.update_td_error(indexes.numpy(), individual_loss)
            # self.model.module.conv1.weight
            if self.alpha_v:
                self.optimizer_v.zero_grad()
                loss_v.backward(retain_graph=True)
                self.optimizer_v.step()

            if self.alpha_q:
                self.optimizer_q.zero_grad()
                loss_q.backward(retain_graph=True)
                self.optimizer_q.step()

            if self.alpha_b:
                self.optimizer_beta.zero_grad()
                loss_b.backward(retain_graph=True)
                self.optimizer_beta.step()

            if self.alpha_r:
                self.optimizer_r.zero_grad()
                loss_r.backward(retain_graph=True)
                self.optimizer_r.step()

            if self.alpha_p:
                self.optimizer_p.zero_grad()
                loss_p.backward()
                self.optimizer_p.step()

            # add results
            results['loss_q'].append(loss_q.data.cpu().numpy()[0])
            results['loss_v'].append(loss_v.data.cpu().numpy()[0])
            results['loss_b'].append(loss_b.data.cpu().numpy()[0])
            results['loss_r'].append(loss_r.data.cpu().numpy()[0])
            results['loss_p'].append(loss_p.data.cpu().numpy()[0])
            results['n'].append(n)

            if not n % self.update_target_interval:
                # self.update_target()
                self.scheduler_v.step()
                self.scheduler_beta.step()
                self.scheduler_q.step()
                self.scheduler_r.step()
                self.scheduler_p.step()

            # if an index is rolled more than once during update_memory_interval period, only the last occurance affect the
            if not (n+1) % self.update_memory_interval and self.prioritized_replay:
                self.train_dataset.update_probabilities()

            # update a global n_step parameter
            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step(n+1)

            if not (n+1) % n_interval:
                yield results
                self.model.train()
                # self.target.eval()
                results = {key: [] for key in results}

    def test_stochastic(self, n_interval, n_tot):

        self.model.eval()
        # self.target.eval()

        results = {'n': [], 'loss_v': [], 'loss_b': [], 'loss_q': [], 'loss_p': [], 'loss_r': [], 'act_diff': [], 'a_agent': [], 'a_player': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)
            a = Variable(sample['a'].cuda(async=True).float().unsqueeze(1), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(async=True).float().unsqueeze(1), requires_grad=False)
            r = Variable(sample['r'].cuda(async=True).float().unsqueeze(1), requires_grad=False)
            t = Variable(sample['t'].cuda(async=True).float().unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True).float().unsqueeze(1), requires_grad=False)
            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)
            f = Variable(sample['f'].cuda(async=True).float().unsqueeze(1), requires_grad=False)
            indexes = sample['i']

            value, q, beta, reward, p, phi = self.model(s, a)
            _, _, _, _, _, phi_tag = self.model(s_tag, a_tag)

            q = q.squeeze(1)
            # m = (((f - value) > 0).float() + 1) if n > self.go_to_max else Variable(torch.ones(f.data.shape).cuda())
            # m = m.detach()

            loss_v = self.alpha_v * self.loss_fn_value(value, f)
            loss_q = self.alpha_q * self.loss_fn_q(q, f)

            # zerovar = Variable(torch.zeros(f.data.shape).cuda(), requires_grad=False)
            # if n > self.go_to_max:
            #     loss_v = self.alpha_v * self.loss_fn_value(F.relu(f - value), zerovar)
            #     loss_q = self.alpha_q * self.loss_fn_q(F.relu(f - q), zerovar)
            # else:
            #     loss_v = self.alpha_v * self.loss_fn_value(value - f, zerovar)
            #     loss_q = self.alpha_q * self.loss_fn_q(q - f, zerovar)
            loss_b = self.alpha_b * self.loss_fn_beta(beta, a_index)

            loss_r = self.alpha_r * self.loss_fn_r(reward, r)

            phi_tag = Variable(phi_tag.data, requires_grad=False)
            loss_p = self.alpha_p * self.loss_fn_p(p, phi_tag)

            # collect actions statistics
            a_index_np = a_index.data.cpu().numpy()

            _, beta_index = beta.data.cpu().max(1)
            beta_index = beta_index.numpy()
            act_diff = (a_index_np != beta_index).astype(np.int)

            # add results
            results['act_diff'].append(act_diff)
            results['a_agent'].append(beta_index)
            results['a_player'].append(a_index_np)
            results['loss_q'].append(loss_q.data.cpu().numpy()[0])
            results['loss_v'].append(loss_v.data.cpu().numpy()[0])
            results['loss_b'].append(loss_b.data.cpu().numpy()[0])
            results['loss_r'].append(loss_r.data.cpu().numpy()[0])
            results['loss_p'].append(loss_p.data.cpu().numpy()[0])
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

    # def get_action_index(self, a):
    #     m = np.zeros((self.global_action_space, self.skip))
    #     m[a, range(self.skip)] = 1
    #     m = m.sum(1)
    #     a = (1 + np.argmax(m[1:])) * (a.sum() != 0)
    #     # transform a to a valid activation
    #     a = self.actions_transform[a]
    #     return a, a

    def learn_deterministic(self, n_interval, n_tot):

        self.model.train()
        # self.target.eval()
        results = {'n': [], 'loss_v': [], 'loss_b': [], 'loss_q': [], 'loss_p': [], 'loss_r': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)
            a = Variable(sample['a'].cuda(async=True), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(async=True), requires_grad=False)
            r = Variable(sample['r'].cuda(async=True).unsqueeze(1), requires_grad=False)
            t = Variable(sample['t'].cuda(async=True).unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True).unsqueeze(1), requires_grad=False)
            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)
            f = Variable(sample['f'].cuda(async=True).unsqueeze(1), requires_grad=False)
            indexes = sample['i']

            value, q, beta, reward, p, phi = self.model(s, a)
            _, _, _, _, _, phi_tag = self.model(s_tag, a_tag)

            m = (((f - value) > 0).float() + 1) if n > self.go_to_max else Variable(torch.ones(f.data.shape).cuda())
            m = m.detach()


            loss_v = self.alpha_v * self.loss_fn_value(value*m, f*m)
            loss_q = self.alpha_q * self.loss_fn_q(q*m, f*m)
            loss_b = self.alpha_b * self.loss_fn_beta(beta * m.repeat(1, 3), a * m.repeat(1, 3))

            loss_r = self.alpha_r * self.loss_fn_r(reward, r)

            phi_tag = Variable(phi_tag.data, requires_grad=False)
            loss_p = self.alpha_p * self.loss_fn_p(p, phi_tag)

            if self.alpha_v:
                self.optimizer_v.zero_grad()
                loss_v.backward(retain_graph=True)
                self.optimizer_v.step()

            if self.alpha_q:
                self.optimizer_q.zero_grad()
                loss_q.backward(retain_graph=True)
                self.optimizer_q.step()

            if self.alpha_b:
                self.optimizer_beta.zero_grad()
                loss_b.backward(retain_graph=True)
                self.optimizer_beta.step()

            if self.alpha_r:
                self.optimizer_r.zero_grad()
                loss_r.backward(retain_graph=True)
                self.optimizer_r.step()

            if self.alpha_p:
                self.optimizer_p.zero_grad()
                loss_p.backward()
                self.optimizer_p.step()

            # add results
            results['loss_q'].append(loss_q.data.cpu().numpy()[0])
            results['loss_v'].append(loss_v.data.cpu().numpy()[0])
            results['loss_b'].append(loss_b.data.cpu().numpy()[0])
            results['loss_r'].append(loss_r.data.cpu().numpy()[0])
            results['loss_p'].append(loss_p.data.cpu().numpy()[0])
            results['n'].append(n)

            if not n % self.update_target_interval:
                # self.update_target()
                self.scheduler_v.step()
                self.scheduler_beta.step()
                self.scheduler_q.step()
                self.scheduler_r.step()
                self.scheduler_p.step()

            # if an index is rolled more than once during update_memory_interval period, only the last occurance affect the
            if not (n+1) % self.update_memory_interval and self.prioritized_replay:
                self.train_dataset.update_probabilities()

            # update a global n_step parameter
            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step(n+1)

            if not (n+1) % n_interval:
                yield results
                self.model.train()
                # self.target.eval()
                results = {key: [] for key in results}

    def test_deterministic(self, n_interval, n_tot):

        self.model.eval()
        # self.target.eval()

        results = {'n': [], 'loss_v': [], 'loss_b': [], 'loss_q': [], 'loss_p': [],
                   'loss_r': [], 'act_diff': [], 'a_agent': [], 'a_player': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)
            a = Variable(sample['a'].cuda(async=True), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(async=True), requires_grad=False)
            r = Variable(sample['r'].cuda(async=True).unsqueeze(1), requires_grad=False)
            t = Variable(sample['t'].cuda(async=True).unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True).unsqueeze(1), requires_grad=False)
            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)
            f = Variable(sample['f'].cuda(async=True).unsqueeze(1), requires_grad=False)
            indexes = sample['i']

            value, q, beta, reward, p, phi = self.model(s, a)
            _, _, _, _, _, phi_tag = self.model(s_tag, a_tag)

            m = (((f - value) > 0).float() + 1) if n > self.go_to_max else Variable(torch.ones(f.data.shape).cuda())
            m = m.detach()

            loss_v = self.alpha_v * self.loss_fn_value(value*m, f*m)
            loss_q = self.alpha_q * self.loss_fn_q(q*m, f*m)

            loss_b = self.alpha_b * self.loss_fn_beta(beta * m.repeat(1,3), a * m.repeat(1,3))

            loss_r = self.alpha_r * self.loss_fn_r(reward, r)

            phi_tag = Variable(phi_tag.data, requires_grad=False)
            loss_p = self.alpha_p * self.loss_fn_p(p, phi_tag)

            # calculate action imitation statistics
            beta_index = (beta.sign().int() * (beta.abs() > 0.5).int()).data.cpu().numpy()
            beta_index[:, 0] = abs(beta_index[:, 0])
            beta_index = np.array([self.reverse_excitation_index[tuple(i)] for i in beta_index])
            a_index_np = a_index.data.cpu().numpy()
            act_diff = (a_index_np != beta_index).astype(np.int)

            # add results
            results['loss_q'].append(loss_q.data.cpu().numpy()[0])
            results['loss_v'].append(loss_v.data.cpu().numpy()[0])
            results['loss_b'].append(loss_b.data.cpu().numpy()[0])
            results['loss_r'].append(loss_r.data.cpu().numpy()[0])
            results['loss_p'].append(loss_p.data.cpu().numpy()[0])
            results['act_diff'].append(act_diff)
            results['a_agent'].append(beta_index)
            results['a_player'].append(a_index_np)


            results['n'].append(n)

            if not (n+1) % n_interval:
                results['s'] = s.data.cpu()
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_agent'] = np.concatenate(results['a_agent'])
                results['a_player'] = np.concatenate(results['a_player'])
                yield results
                self.model.eval()
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

    def play_episode_stochastic(self, n_tot):

        self.model.eval()
        env = Env()

        n_human = 300
        humans_trajectories = iter(self.data)
        softmax = torch.nn.Softmax()

        # self.actions_matrix = torch.FloatTensor([[0, 0, 0], [1, 0, 0],[0, 1, 0], [0, 0, 1]])

        for i in range(n_tot):

            env.reset()
            observation = next(humans_trajectories)
            trajectory = self.data[observation]
            choices = np.arange(self.global_action_space, dtype=np.int)

            j = 0

            while not env.t:

                s = Variable(env.s.cuda(), requires_grad=False)
                v, q, beta, _, _, phi = self.model(s, self.actions_matrix)
                beta = beta.squeeze(0)
                q = q.squeeze(2)
                q = q.squeeze(0)
                # beta[0] = 0

                if self.imitation:
                    pi = (beta > 5).float() * (q / q.max())
                else:
                    pi = q / q.max() # q.max() is the temperature

                beta_prob = softmax(pi)

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:
                    # a = np.random.choice(choices)
                    if self.greedy:
                        a = pi.data.cpu().numpy()
                        a = np.argmax(a)
                    else:
                        a = softmax(pi).data.cpu().numpy()
                        a = np.random.choice(choices, p=a)

                env.step(a)

                # x = phi.squeeze(0).data.cpu().numpy()
                # print(np.mean(abs(x)))
                # yield v, q, beta, r, p, s
                yield {'o': env.s.cpu().numpy(),
                       'v': v.data.cpu().numpy(),
                       's': phi.data.cpu().numpy(),
                       'score': env.score,
                       'beta': beta_prob.data.cpu().numpy(),
                       'phi': phi.squeeze(0).data.cpu().numpy()}

                j += 1

        raise StopIteration

    def play_episode_deterministic(self, n_tot):
        self.model.eval()
        env = Env()

        n_human = 300
        humans_trajectories = iter(self.data)
        reverse_excitation_index = consts.reverse_excitation_index

        for i in range(n_tot):

            env.reset()
            observation = next(humans_trajectories)
            trajectory = self.data[observation]

            j = 0

            while not env.t:

                s = Variable(env.s.cuda(), requires_grad=False)
                v, q, beta, r, p, phi = self.model(s)
                beta = beta.squeeze(0)

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:

                    beta_index = (beta.sign().int() * (beta.abs() > 0.5).int()).data.cpu().numpy()
                    beta_index[0] = abs(beta_index[0])
                    a = reverse_excitation_index[tuple(beta_index.data)]

                env.step(a)

                # x = phi.squeeze(0).data.cpu().numpy()
                # print(np.mean(abs(x)))
                # yield v, q, beta, r, p, s
                yield {'o': env.s.cpu().numpy(),
                       'v': v.data.cpu().numpy(),
                       's': phi.data.cpu().numpy(),
                       'score': env.score,
                       'beta': beta.data.cpu().numpy(),
                       'phi': phi.squeeze(0).data.cpu().numpy()}

                j += 1

        raise StopIteration

    def play_deterministic(self, n_tot):

        self.model.eval()
        env = Env()
        render = args.render

        n_human = 60
        humans_trajectories = iter(self.data)
        reverse_excitation_index = consts.reverse_excitation_index

        for i in range(n_tot):

            env.reset()

            observation = next(humans_trajectories)
            print("Observation %s" % observation)
            trajectory = self.data[observation]

            j = 0

            while not env.t:

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:


                    if self.cuda:
                        s = Variable(env.s.cuda(), requires_grad=False)
                    else:
                        s = Variable(env.s, requires_grad=False)
                    _, _, beta, _, _, _ = self.model(s)

                    beta = beta.squeeze(0)
                    beta = (beta.sign().int() * (beta.abs() > 0.5).int()).data
                    if self.cuda:
                        beta = beta.cpu().numpy()
                    else:
                        beta = beta.numpy()
                    beta[0] = abs(beta[0])
                    a = reverse_excitation_index[tuple(beta)]

                env.step(a)

                j += 1

            yield {'o': env.s.cpu().numpy(),
                   'score': env.score}