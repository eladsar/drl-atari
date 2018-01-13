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
from model import BehavioralRNN
from memory_rnn import DemonstrationRNNMemory, DemonstrationRNNBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationRNNSampler
from agent import Agent
from environment import Env

class BehavioralAgent(Agent):

    def __init__(self, load_dataset=True):

        super(BehavioralAgent, self).__init__()

        self.actions_transform = np.array(consts.action2activation[args.game])

        if load_dataset:
            # demonstration source
            self.meta, self.data = preprocess_demonstrations()
            self.meta = divide_dataset(self.meta)

            # datasets
            self.train_dataset = DemonstrationRNNMemory("train", self.meta, self.data)
            self.val_dataset = DemonstrationRNNMemory("val", self.meta, self.data)
            self.test_dataset = DemonstrationRNNMemory("test", self.meta, self.data)
            self.full_dataset = DemonstrationRNNMemory("full", self.meta, self.data)

            self.train_sampler = DemonstrationRNNBatchSampler(self.train_dataset, train=True)
            self.val_sampler = DemonstrationRNNBatchSampler(self.train_dataset, train=False)
            self.test_sampler = DemonstrationRNNBatchSampler(self.test_dataset, train=False)
            self.episodic_sampler = SequentialDemonstrationRNNSampler(self.full_dataset)

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
            self.model_single = BehavioralNetDeterministic(self.global_action_space)

        else:
            self.alpha_q = 1  # 1 / 0.02
            self.loss_fn_beta = torch.nn.CrossEntropyLoss()
            self.learn = self.learn_stochastic
            self.test = self.test_stochastic
            self.play = self.play_stochastic
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
        excitation = torch.LongTensor(consts.excitation_map)
        mask = torch.LongTensor(consts.excitation_mask[args.game])
        mask_dup = mask.unsqueeze(0).repeat(consts.action_space, 1)
        actions = Variable(mask_dup * excitation, requires_grad=False)
        if args.cuda:
            actions = actions.cuda()

        self.actions_matrix = actions.unsqueeze(0)
        self.actions_matrix = self.actions_matrix.repeat(1, 1, 1).float()

        self.go_to_max = np.inf # 4096 * 8 * 4

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

        results = {'n': [], 'loss_v': [], 'loss_b': [], 'loss_q': [], 'loss_p': [], 'loss_r': []}

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

            # add results
            results['loss_q'].append(loss_q.data.cpu().numpy()[0])
            results['loss_v'].append(loss_v.data.cpu().numpy()[0])
            results['loss_b'].append(loss_b.data.cpu().numpy()[0])
            results['loss_r'].append(loss_r.data.cpu().numpy()[0])
            results['loss_p'].append(loss_p.data.cpu().numpy()[0])
            results['n'].append(n)

            if not (n+1) % n_interval:
                yield results
                self.model.eval()
                # self.target.eval()
                results = {key: [] for key in results}


    def play_deterministic_episode(self, n_tot):
        self.model.eval()
        env = Env()

        n_human = 60
        humans_trajectories = iter(self.data)
        reverse_excitation_index = consts.reverse_excitation_index

        for i in range(n_tot):

            env.reset()
            observation = next(humans_trajectories)
            trajectory = self.data[observation]

            j = 0

            while not env.t:

                if self.cuda:
                    s = Variable(env.s.cuda(), requires_grad=False)
                else:
                    s = Variable(env.s, requires_grad=False)
                v, q, beta, r, p, phi = self.model(s)

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:

                    beta = beta.squeeze(0)
                    beta = beta.sign().int() * (beta.abs() > 0.5).int()
                    a = reverse_excitation_index[tuple(beta.data)]

                env.step(a)
                # yield v, q, beta, r, p, s
                yield {'o': env.s.cpu().numpy(),
                       'v': v.data.cpu().numpy(),
                       's': phi.data.cpu().numpy(),
                       'score': env.score}

                j += 1
                print(j)

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

            ims = []
            # fig = plt.figure()
            while not env.t:

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:

                    # im = plt.imshow(np.rollaxis(env.s.numpy().squeeze(0)[:3], 0, 3), animated=True)
                    # ims.append([im])
                    if self.cuda:
                        s = Variable(env.s.cuda(), requires_grad=False)
                    else:
                        s = Variable(env.s, requires_grad=False)
                    _, _, beta, _, _, _ = self.model(s)

                    beta = beta.squeeze(0)
                    beta = beta.sign().int() * (beta.abs() > 0.5).int()
                    a = reverse_excitation_index[tuple(beta.data)]

                env.step(a)

                j += 1

            # if render:
            #     ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
            #                                     repeat=False)
            #     plt.show()

            yield env.score