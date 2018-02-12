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
from model import BehavioralHotNet
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationSampler
from agent import Agent
from environment import Env


class BehavioralHotAgent(Agent):

    def __init__(self, load_dataset=True):

        super(BehavioralHotAgent, self).__init__()

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
            self.loss_fn_q = torch.nn.L1Loss(size_average=True)
        else:
            self.loss_fn_value = torch.nn.MSELoss(size_average=True)
            self.loss_fn_q = torch.nn.MSELoss(size_average=True)

        self.loss_fn_r = torch.nn.MSELoss(size_average=True)
        self.loss_fn_p = torch.nn.L1Loss(size_average=True)

        if self.weight_by_expert:
            self.loss_fn_beta = torch.nn.CrossEntropyLoss(reduce=False)
        else:
            self.loss_fn_beta = torch.nn.CrossEntropyLoss(reduce=True)


        # alpha weighted sum

        self.alpha_v = 1  # 1 / 0.02
        self.alpha_b = 1  # 1 / 0.7

        self.alpha_r = 1  # 1 / 0.7
        self.alpha_p = 0  # 1 / 0.7
        self.alpha_q = 1

        self.model = BehavioralHotNet()
        self.model.cuda()

        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_v = BehavioralHotAgent.set_optimizer(self.model.parameters(), args.lr)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v, self.decay)

        self.optimizer_beta = BehavioralHotAgent.set_optimizer(self.model.parameters(), args.lr_beta)
        self.scheduler_beta = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_beta, self.decay)

        self.optimizer_q = BehavioralHotAgent.set_optimizer(self.model.parameters(), args.lr_q)
        self.scheduler_q = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q, self.decay)

        self.optimizer_r = BehavioralHotAgent.set_optimizer(self.model.parameters(), args.lr_r)
        self.scheduler_r = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_r, self.decay)

        self.optimizer_p = BehavioralHotAgent.set_optimizer(self.model.parameters(), args.lr_p)
        self.scheduler_p = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_p, self.decay)

        self.episodic_evaluator = self.dummy_episodic_evaluator

        actions = torch.FloatTensor(consts.hotvec_matrix) / (3**(0.5))
        actions = Variable(actions, requires_grad=False).cuda()

        self.actions_matrix = actions.unsqueeze(0)
        # self.reverse_excitation_index = consts.hotvec_inv

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

    def learn(self, n_interval, n_tot):

        self.model.train()
        # self.target.eval()
        results = {'n': [], 'loss_v': [], 'loss_b': [], 'loss_q': [], 'loss_p': [], 'loss_r': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda(), requires_grad=False)
            a_tag = Variable(sample['a_tag'].float().cuda(), requires_grad=False)
            r = Variable(sample['r'].cuda().unsqueeze(1), requires_grad=False)
            t = Variable(sample['t'].cuda().unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda().unsqueeze(1), requires_grad=False)
            a_index = Variable(sample['a_index'].cuda(), requires_grad=False)
            f = Variable(sample['f'].cuda().unsqueeze(1), requires_grad=False)
            score = Variable(sample['score'].cuda(async=True).unsqueeze(1), requires_grad=False)
            w = Variable(sample['w'].cuda(), requires_grad=False)

            indexes = sample['i']

            value, q, beta, reward, p, phi = self.model(s, a)

            _, _, _, _, _, phi_tag = self.model(s_tag, a_tag)

            loss_v = self.alpha_v * self.loss_fn_value(value, f + self.final_score_reward * score)
            loss_q = self.alpha_q * self.loss_fn_q(q, f + self.final_score_reward * score)

            if self.weight_by_expert:
                loss_b = self.alpha_b * (self.loss_fn_beta(beta, a_index) * w).sum() / self.batch
            else:
                loss_b = self.alpha_b * self.loss_fn_beta(beta, a_index)

            loss_r = self.alpha_r * self.loss_fn_r(reward, r)

            phi_tag = phi_tag.detach()
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

                # param = self.model.fc_z_p.bias.data.cpu().numpy()
                # if np.isnan(param.max()):
                #     print("XXX")
                # paramgrad = self.model.fc_z_p.bias.grad.data.cpu().numpy()
                # param_l = loss_q.data.cpu().numpy()[0]
                # print("max: %g | min: %g | max_grad: %g | min_grad: %g | loss: %g " % (param.max(), param.min(), paramgrad.max(), paramgrad. min(), param_l))


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

    def test(self, n_interval, n_tot):

        self.model.eval()
        # self.target.eval()

        results = {'n': [], 'loss_v': [], 'loss_b': [], 'loss_q': [], 'loss_p': [], 'loss_r': [], 'act_diff': [], 'a_agent': [], 'a_player': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(async=True), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(async=True), requires_grad=False)
            a = Variable(sample['a'].cuda(async=True).unsqueeze(1), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(async=True).unsqueeze(1), requires_grad=False)
            r = Variable(sample['r'].cuda(async=True).unsqueeze(1), requires_grad=False)
            t = Variable(sample['t'].cuda(async=True).unsqueeze(1), requires_grad=False)
            k = Variable(sample['k'].cuda(async=True).unsqueeze(1), requires_grad=False)
            a_index = Variable(sample['a_index'].cuda(async=True), requires_grad=False)
            score = Variable(sample['score'].cuda(async=True).unsqueeze(1), requires_grad=False)
            w = Variable(sample['w'].cuda(), requires_grad=False)

            f = Variable(sample['f'].cuda(async=True).unsqueeze(1), requires_grad=False)
            indexes = sample['i']

            value, q, beta, reward, p, phi = self.model(s, a)
            _, _, _, _, _, phi_tag = self.model(s_tag, a_tag)

            q = q.squeeze(1)
            reward = reward.squeeze(1)

            loss_v = self.alpha_v * self.loss_fn_value(value, f + self.final_score_reward * score)
            loss_q = self.alpha_q * self.loss_fn_q(q, f + self.final_score_reward * score)

            if self.weight_by_expert:
                loss_b = self.alpha_b * (self.loss_fn_beta(beta, a_index) * w).sum() / self.batch
            else:
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

        mask = torch.FloatTensor(consts.actions_mask[args.game])
        mask = Variable(mask.cuda(), requires_grad=False)
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

                q = q * mask
                # beta[0] = 0
                temp = 0.1
                if True: # self.imitation:

                    # consider only 3 most frequent actions
                    beta_np = beta.data.cpu().numpy()
                    indices = np.argsort(beta_np)

                    # maskb = Variable(torch.FloatTensor([i in indices[14:18] for i in range(18)]), requires_grad=False)
                    # maskb = Variable(torch.FloatTensor([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), requires_grad=False)
                    # maskb = maskb.cuda()
                    # pi = maskb * (q / q.max())

                    maskb = Variable(torch.FloatTensor([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), requires_grad=False)
                    maskb = maskb.cuda()
                    pi = maskb * (beta / beta.max())
                    # pi = maskb * (q / q.max())
                    self.greedy = False

                    # if j%2:
                    #     pi = maskb * (q / q.max())
                    #     self.greedy = True
                    # else:
                    #     self.greedy = False
                    #     pi = maskb * (beta / beta.max())
                    # pi = (beta > 3).float() * (q / q.max())

                    # pi = beta  # (beta > 5).float() * (q / q.max())
                    # pi[0] = 0
                    # beta_prob = softmax(pi)
                    beta_prob = pi
                else:
                    pi = q / q.max() # q.max() is the temperature
                    beta_prob = q

                if j < n_human:
                    a = trajectory[j, self.meta['action']]

                else:
                    # a = np.random.choice(choices)
                    if self.greedy:
                        a = pi.data.cpu().numpy()
                        a = np.argmax(a)
                    else:
                        a = softmax(pi/temp).data.cpu().numpy()
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
