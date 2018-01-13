import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from config import consts, args
from model import DQN, DQNDueling, DVAN_ActionIn, DVAN_ActionOut
from memory import DemonstrationMemory, DemonstrationBatchSampler, \
     preprocess_demonstrations, divide_dataset, \
     SequentialDemonstrationSampler
from player import player_worker, QPlayer, AVPlayer, AVAPlayer

class Agent(object):

    def __init__(self):
        self.model = None
        self.optimizer = None
        # parameters
        self.margin = args.margin
        self.batch = args.batch
        self.on_policy = args.on_policy
        self.double_q = args.double_q
        self.dueling = args.dueling
        self.prioritized_replay = args.prioritized_replay
        self.discount = args.discount
        self.update_target_interval = args.update_target_interval
        self.update_n_steps_interval = args.update_n_steps_interval
        self.update_memory_interval = args.update_memory_interval
        self.value_advantage = args.value_advantage
        self.action_space = consts.n_actions[args.game]
        self.global_action_space = consts.action_space
        self.decay = args.decay
        self.myopic = args.myopic
        self.l1_loss = args.l1_loss
        self.cuda = args.cuda
        self.skip = args.skip
        self.greedy = args.greedy
        self.imitation = args.imitation
        self.final_score_reward = args.final_score_reward

    @staticmethod
    def set_optimizer(params, lr=args.lr):

        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr,)

        elif args.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=0.01, weight_decay=0, momentum=0.1, centered=True)

        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, betas=(args.adam_beta1, args.adam_beta1), eps=args.adam_eps)

        return optimizer

    def save_checkpoint(self, path, aux=None):

        state = {'state_dict': self.model.state_dict(),
                 'optimizer_dict': self.optimizer.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer_dict'])

        return state['aux']

    def train(self, n_interval, n_tot):
        raise NotImplementedError

    def evaluate(self, n_interval, n_tot):
        raise NotImplementedError

    def play(self, n_interval, n_tot):
        raise NotImplementedError

    def resume(self, model_path):
        raise NotImplementedError

    @staticmethod
    def individual_loss_fn(argument):
        raise NotImplementedError


class LfdAgent(Agent):

    def __init__(self):

        super(LfdAgent, self).__init__()

        # demonstration source
        self.meta, self.data = preprocess_demonstrations()
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

        # set learn validate test play parameters based on arguments
        # configure learning
        if not args.value_advantage:
            self.learn = self.learn_q
            self.test = self.test_q
            self.player = QPlayer
            self.agent_type = 'q'
            # loss function and optimizer

            if self.l1_loss:
                self.loss_fn = torch.nn.L1Loss(size_average=True)
                self.individual_loss_fn = self.individual_loss_fn_l1
            else:
                self.loss_fn = torch.nn.MSELoss(size_average=True)
                self.individual_loss_fn = self.individual_loss_fn_l2

            # Choose a model acording to the configurations
            models = {(0,): DQN, (1,): DQNDueling}
            Model = models[(self.dueling,)]

            self.model_single = Model(self.action_space)
            self.target_single = Model(self.action_space)

        else:

            if args.value_only:
                self.alpha_v, self.alpha_a = 1, 0
            else:
                self.alpha_v, self.alpha_a = 1, 1

            if self.l1_loss:
                self.loss_fn_value = torch.nn.L1Loss(size_average=True)
                self.loss_fn_advantage = torch.nn.L1Loss(size_average=True)
                self.individual_loss_fn = self.individual_loss_fn_l1
            else:
                self.loss_fn_value = torch.nn.MSELoss(size_average=True)
                self.loss_fn_advantage = torch.nn.MSELoss(size_average=True)
                self.individual_loss_fn = self.individual_loss_fn_l2

            if not args.input_actions:
                self.learn = self.learn_va
                self.test = self.test_va
                self.player = AVPlayer
                self.agent_type = 'av'
                self.model_single = DVAN_ActionOut(self.action_space)
                self.target_single = DVAN_ActionOut(self.action_space)

            else:
                self.learn = self.learn_ava
                self.test = self.test_ava
                self.player = AVAPlayer
                self.agent_type = 'ava'
                self.model_single = DVAN_ActionIn(3)
                self.target_single = DVAN_ActionIn(3)

                # model specific parameters
                self.action_space = consts.action_space
                self.excitation = torch.LongTensor(consts.excitation_map)
                self.excitation_length = self.excitation.shape[0]
                self.mask = torch.LongTensor(consts.excitation_mask[args.game])
                self.mask_dup = self.mask.unsqueeze(0).repeat(self.action_space, 1)

                actions = Variable(self.mask_dup * self.excitation, requires_grad=False)
                actions = actions.cuda()

                self.actions_matrix = actions.unsqueeze(0)
                self.actions_matrix = self.actions_matrix.repeat(self.batch, 1, 1).float()
                self.reverse_excitation_index = consts.reverse_excitation_index

        if not args.play:
            self.play = self.dummy_play
        elif args.gpu_workers == 0:
            self.play = self.single_play
        else:
            self.play = self.multi_play

        q_functions = {(0, 0): self.simple_q, (0, 1): self.double_q, (1, 0): self.simple_on_q,
                       (1, 1): self.simple_on_q}
        self.q_estimator = q_functions[(self.double_q, self.on_policy)]

        # configure learning
        if args.cuda:
            self.model_single = self.model_single.cuda()
            self.target_single = self.target_single.cuda()
            self.model = torch.nn.DataParallel(self.model_single)
            self.target = torch.nn.DataParallel(self.target_single)
        else:
            self.model = self.model_single
            self.target = self.target_single
        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer = LfdAgent.set_optimizer(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.decay)

    @staticmethod
    def individual_loss_fn_l2(argument):
        return abs(argument.data.cpu().numpy())**2

    def individual_loss_fn_l1(argument):
        return abs(argument.data.cpu().numpy())

    def resume(self, model_path):

        aux = self.load_checkpoint(model_path)
        self.update_target()
        return aux

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    # double q-learning
    def double_q(self, s_tag, q_target, a_tag, f):
        q_tag = self.model(s_tag)
        _, a_tag = q_tag.max(1)
        return q_target[range(self.batch), a_tag.data].data

    def simple_q(self, s_tag, q_target, a_tag, f):
        return q_target.max(1)[0].data

    def simple_on_q(self, s_tag, q_target, a_tag, f):
        return q_target[range(self.batch), a_tag.data].data

    def complex_q(self, s_tag, q_target, a_tag, f):
        q_tag = self.model(s_tag)

        le = self.margin * Variable(torch.ones(self.batch, self.action_space).cuda())
        le[range(self.batch), a_tag.data] = 0

        _, a_max = (q_tag + le).max(1)
        off_policy = q_target[range(self.batch), a_max.data]

        on_policy = q_target[range(self.batch), a_tag.data]
        alpha = torch.clamp(f / 2, 0, 1)  # * np.exp(- (n / (32 * n_interval)))

        return torch.mul(on_policy.data, alpha) + torch.mul((1 - alpha), off_policy.data)

    def dummy_episodic_evaluator(self):
        while True:
            yield {'q_diff': torch.zeros(100), 'a_agent': torch.zeros(100, self.action_space), 'a_player': torch.zeros(100).long()}

    def episodic_evaluator(self):

        self.model.eval()
        self.target.eval()
        results = {'q_diff': [], 'a_agent': [], 'a_player': []}

        for n, sample in tqdm(enumerate(self.episodic_loader)):

            is_final = sample['is_final']
            final_indicator, final_index = is_final.max(0)
            final_indicator = final_indicator.numpy()[0]
            final_index = final_index.numpy()[0]+1 if final_indicator else self.batch

            s = Variable(sample['s'][:final_index].cuda(), requires_grad=False)
            a = Variable(sample['a'][:final_index].cuda().unsqueeze(1), requires_grad=False)
            f = sample['f'][:final_index].float().cuda()
            base = sample['base'][:final_index].float()
            r = Variable(sample['r'][:final_index].float().cuda(), requires_grad=False)
            a_index = Variable(sample['a_index'][:final_index].cuda().unsqueeze(1), requires_grad=False)

            if self.agent_type == 'q':
                q = self.model(s)
                q_best, a_best = q.cpu().data.max(1)
                q_diff = r.data.cpu() - q_best if self.myopic else (f.cpu() - base) - q_best
                a_player = a.squeeze(1).data.cpu()
                a_agent = q.data.cpu()

            else:

                if self.agent_type == 'av':
                    value, advantage = self.model(s)
                elif self.agent_type == 'ava':
                    value, advantage = self.model(s, self.actions_matrix[:final_index])
                else:
                    raise NotImplementedError
                value = value.squeeze(1)
                q_diff = r.data.cpu() - value.data.cpu() if self.myopic else (f.cpu() - base) - value.data.cpu()
                a_player = a_index.squeeze(1).data.cpu()
                a_agent = advantage.data.cpu()

            # add results
            results['q_diff'].append(q_diff)
            results['a_agent'].append(a_agent)
            results['a_player'].append(a_player)

            if final_indicator:
                results['q_diff'] = torch.cat(results['q_diff'])
                results['a_agent'] = torch.cat(results['a_agent'])
                results['a_player'] = torch.cat(results['a_player'])
                yield results
                self.model.eval()
                self.target.eval()
                results = {key: [] for key in results}

    def saliency_map(self, s, a):
        self.model.eval()
        pass

    def learn_q(self, n_interval, n_tot):

        self.model.train()
        self.target.eval()
        results = {'loss': [], 'n': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)
            k = Variable(sample['k'].float().cuda(), requires_grad=False)
            f = sample['f'].float().cuda()
            indexes = sample['i']

            q = self.model(s)
            q_a = q.gather(1, a)[:, 0]

            q_target = self.target(s_tag)

            max_q_target = Variable(self.q_estimator(s_tag, q_target, a_tag, f), requires_grad=False)
            loss = self.loss_fn(q_a, r + (self.discount ** k) * (max_q_target * (1 - t)))

            # calculate the td error for the priority replay
            if self.prioritized_replay:
                argument = r + (self.discount ** k) * (max_q_target * (1 - t)) - q_a
                individual_loss = LfdAgent.individual_loss_fn(argument)
                self.train_dataset.update_td_error(indexes.numpy(), individual_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)

            if not n % self.update_target_interval:
                self.update_target()
                self.scheduler.step()

            # if an index is rolled more than once during update_memory_interval period, only the last occurance affect the
            if not (n+1) % self.update_memory_interval and self.prioritized_replay:
                self.train_dataset.update_probabilities()

            # update a global n_step parameter
            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step(n+1)

            if not (n+1) % n_interval:
                results['n_steps'] = self.train_dataset.n_steps
                yield results
                self.model.train()
                self.target.eval()
                results = {key: [] for key in results}

    def learn_ava(self, n_interval, n_tot):

        self.model.train()
        self.target.train()
        results = {'loss': [], 'n': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda(), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)
            k = Variable(sample['k'].float().cuda(), requires_grad=False)
            f = sample['f'].float().cuda()
            indexes = sample['i']

            value, advantage = self.model(s, a)
            value_tag = self.target(s_tag)

            value_target = self.target(s)
            value_tag = Variable(value_tag.data, requires_grad=False)
            value_target = Variable(value_target.data, requires_grad=False)

            value = value.squeeze(1)
            value_tag = value_tag.squeeze(1)
            advantage = advantage.squeeze(1)
            value_target = value_target.squeeze(1)
            loss_v = self.loss_fn_value(value, r + (self.discount ** k) * (value_tag * (1 - t)))
            loss_a = self.loss_fn_advantage(advantage, r + (self.discount ** k) * (value_tag * (1 - t)) - value_target)
            loss = self.alpha_v * loss_v + self.alpha_a * loss_a

            # # calculate the td error for the priority replay
            # if self.prioritized_replay:
            #     argument = r + (self.discount ** k) * (max_q_target * (1 - t)) - q_a
            #     individual_loss = LfdAgent.individual_loss_fn(argument)
            #     self.train_dataset.update_td_error(indexes.numpy(), individual_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)

            if not n % self.update_target_interval:
                self.update_target()
                self.scheduler.step()

            # if an index is rolled more than once during update_memory_interval period,
            # only the last occurrence affect the
            if not (n+1) % self.update_memory_interval and self.prioritized_replay:
                self.train_dataset.update_probabilities()

            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step(n+1)

            if not (n+1) % n_interval:
                results['n_steps'] = self.train_dataset.n_steps
                yield results
                self.model.train()
                self.target.eval()
                results = {key: [] for key in results}

    def learn_va(self, n_interval, n_tot):

        self.model.train()
        self.target.eval()
        results = {'loss': [], 'n': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)
            k = Variable(sample['k'].float().cuda(), requires_grad=False)
            f = sample['f'].float().cuda()
            indexes = sample['i']

            value, advantage = self.model(s)
            value_tag, advantage_tag = self.target(s_tag)
            advantage_a = advantage.gather(1, a)[:, 0]

            value_target, advantage_target = self.target(s)
            value_tag = Variable(value_tag.data, requires_grad=False)
            value_target = Variable(value_target.data, requires_grad=False)

            value = value.squeeze(1)
            value_tag = value_tag.squeeze(1)
            value_target = value_target.squeeze(1)
            loss_v = self.loss_fn_value(value, r + (self.discount ** k) * (value_tag * (1 - t)))
            loss_a = self.loss_fn_advantage(advantage_a, r + (self.discount ** k) * (value_tag * (1 - t)) - value_target)
            loss = self.alpha_v * loss_v + self.alpha_a * loss_a

            # # calculate the td error for the priority replay
            # if self.prioritized_replay:
            #     argument = r + (self.discount ** k) * (max_q_target * (1 - t)) - q_a
            #     individual_loss = LfdAgent.individual_loss_fn(argument)
            #     self.train_dataset.update_td_error(indexes.numpy(), individual_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)

            if not n % self.update_target_interval:
                self.update_target()
                self.scheduler.step()

            # if an index is rolled more than once during update_memory_interval period,
            # only the last occurrence affect the
            if not (n+1) % self.update_memory_interval and self.prioritized_replay:
                self.train_dataset.update_probabilities()

            if not (n+1) % self.update_n_steps_interval:
                self.train_dataset.update_n_step(n+1)

            if not (n+1) % n_interval:
                results['n_steps'] = self.train_dataset.n_steps
                yield results
                self.model.train()
                self.target.eval()
                results = {key: [] for key in results}

    def test_va(self, n_interval, n_tot):

        self.model.eval()
        self.target.eval()

        results = {'loss': [], 'n': [], 'q_diff': [], 'q': [], 'act_diff': [], 'r': [], 'a_best': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)
            k = Variable(sample['k'].float().cuda(), requires_grad=False)
            f = sample['f'].float().cuda()
            base = sample['base'].float()

            value, advantage = self.model(s)
            value_tag, advantage_tag = self.target(s_tag)
            advantage_a = advantage.gather(1, a)[:, 0]

            value_target, advantage_target = self.target(s)
            value_tag = Variable(value_tag.data, requires_grad=False)
            value_target = Variable(value_target.data, requires_grad=False)

            advantage_best, a_best = advantage.data.cpu().max(1)

            value = value.squeeze(1)
            value_tag = value_tag.squeeze(1)
            value_target = value_target.squeeze(1)
            loss_v = self.loss_fn_value(value, r + (self.discount ** k) * (value_tag * (1 - t)))
            loss_a = self.loss_fn_advantage(advantage_a, r + (self.discount ** k) * (value_tag * (1 - t)) - value_target)
            loss = self.alpha_v * loss_v + self.alpha_a * loss_a

            if self.myopic:
                v_diff = r.data.cpu() - value.data.cpu()
            else:
                v_diff = (f.cpu() - base) - value.data.cpu()

            act_diff = (a.squeeze(1).data.cpu() == a_best).numpy()

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)
            results['q_diff'].append(v_diff)
            results['q'].append(advantage.data.cpu())
            results['act_diff'].append(act_diff)
            results['r'].append(r.data.cpu())
            results['a_best'].append(a_best)

            if not (n+1) % n_interval:
                results['s'] = s.data.cpu()
                results['s_tag'] = s_tag.data.cpu()
                results['q'] = torch.cat(results['q'])
                results['r'] = torch.cat(results['r'])
                results['q_diff'] = torch.cat(results['q_diff'])
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_best'] = torch.cat(results['a_best'])
                results['n_steps'] = self.test_dataset.n_steps
                yield results
                self.model.eval()
                self.target.eval()
                results = {key: [] for key in results}

    def test_ava(self, n_interval, n_tot):

        self.model.eval()
        self.target.eval()

        results = {'loss': [], 'n': [], 'q_diff': [], 'q': [], 'act_diff': [], 'r': [], 'a_best': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda(), requires_grad=False)
            a_index = Variable(sample['a_index'].cuda().unsqueeze(1), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)
            k = Variable(sample['k'].float().cuda(), requires_grad=False)
            f = sample['f'].float().cuda()
            base = sample['base'].float()

            # value, advantage = self.eval_ava(s, a)
            value, advantage = self.model(s, self.actions_matrix)
            value_tag = self.target(s_tag)

            value_target = self.target(s)
            value_tag = Variable(value_tag.data, requires_grad=False)
            value_target = Variable(value_target.data, requires_grad=False)

            value = value.squeeze(1)
            value_tag = value_tag.squeeze(1)

            advantage = advantage.squeeze(2)
            advantage_a = advantage.gather(1, a_index)[:, 0]

            value_target = value_target.squeeze(1)

            loss_v = self.loss_fn_value(value, r + (self.discount ** k) * (value_tag * (1 - t)))
            loss_a = self.loss_fn_advantage(advantage_a, r + (self.discount ** k) * (value_tag * (1 - t)) - value_target)
            loss = self.alpha_v * loss_v + self.alpha_a * loss_a

            advantage_best, a_best = advantage.data.cpu().max(1)

            if self.myopic:
                v_diff = r.data.cpu() - value.data.cpu()
            else:
                v_diff = (f.cpu() - base) - value.data.cpu()

            act_diff = (a_index.squeeze(1).data.cpu() == a_best).numpy()

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)
            results['q_diff'].append(v_diff)
            results['q'].append(advantage.data.cpu())
            results['act_diff'].append(act_diff)
            results['r'].append(r.data.cpu())
            results['a_best'].append(a_best)

            if not (n+1) % n_interval:
                results['s'] = s.data.cpu()
                results['s_tag'] = s_tag.data.cpu()
                results['q'] = torch.cat(results['q'])
                results['r'] = torch.cat(results['r'])
                results['q_diff'] = torch.cat(results['q_diff'])
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_best'] = torch.cat(results['a_best'])
                results['n_steps'] = self.test_dataset.n_steps
                yield results
                self.model.eval()
                self.target.eval()
                results = {key: [] for key in results}

    def test_q(self, n_interval, n_tot):

        self.model.eval()
        self.target.eval()

        results = {'loss': [], 'n': [], 'q_diff': [], 'q': [], 'act_diff': [], 'r': [], 'a_best': []}

        for n, sample in tqdm(enumerate(self.test_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
            a_tag = Variable(sample['a_tag'].cuda(), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)
            k = Variable(sample['k'].float().cuda(), requires_grad=False)
            f = sample['f'].float().cuda()
            base = sample['base'].float()

            q = self.model(s)
            q_a = q.gather(1, a)[:, 0]
            q_best, a_best = q.data.cpu().max(1)

            q_target = self.target(s_tag)

            max_q_target = Variable(self.q_estimator(s_tag, q_target, a_tag, f), requires_grad=False)
            loss = self.loss_fn(q_a, r + (self.discount ** k) * (max_q_target * (1 - t)))

            if self.myopic:
                q_diff = r.data.cpu() - q_best
            else:
                q_diff = (f.cpu() - base) - q_best

            act_diff = (a.squeeze(1).data.cpu() == a_best).numpy()

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)
            results['q_diff'].append(q_diff)
            results['q'].append(q.data.cpu())
            results['act_diff'].append(act_diff)
            results['r'].append(r.data.cpu())
            results['a_best'].append(a_best)

            if not (n+1) % n_interval:
                results['s'] = s.data.cpu()
                results['s_tag'] = s_tag.data.cpu()
                results['q'] = torch.cat(results['q'])
                results['r'] = torch.cat(results['r'])
                results['q_diff'] = torch.cat(results['q_diff'])
                results['act_diff'] = np.concatenate(results['act_diff'])
                results['a_best'] = torch.cat(results['a_best'])
                results['n_steps'] = self.test_dataset.n_steps
                yield results
                self.model.eval()
                self.target.eval()
                results = {key: [] for key in results}

    def dummy_play(self, n_interval, n_tot):
        while True:
            yield {'scores': [0] * n_interval, 'epoch': range(n_interval)}

    def single_play(self, n_interval, n_tot):

        player = self.player()
        results = {'scores': [], 'epoch': []}

        for epoch in range(0, n_tot, n_interval):

            params = self.model.state_dict()
            for i in tqdm(range(n_interval)):
                score = player.play(params)
                results['scores'].append(score)
                results['epoch'].append(epoch + i)

            yield results
            results = {key: [] for key in results}

    def multi_play(self, n_interval, n_tot):

        ctx = mp.get_context('forkserver')
        queue = ctx.Queue()
        # new = mp.Event()
        jobs = ctx.Queue(n_interval)
        done = ctx.Event()

        processes = []
        for rank in range(args.gpu_workers):
            p = ctx.Process(target=player_worker, args=(queue, jobs, done, self.player))
            p.start()
            processes.append(p)

        try:

            results = {'scores': [], 'epoch': []}

            for epoch in range(0, n_tot, n_interval):

                params = self.model.state_dict()
                [jobs.put(params) for i in range(n_interval)]

                for i in tqdm(range(n_interval)):
                    score = queue.get()
                    results['scores'].append(score)
                    results['epoch'].append(epoch + i)

                yield results
                results = {key: [] for key in results}

            raise StopIteration

        finally:

            done.set()
            for p in processes:
                p.join()



    # def multi_play(self, n_interval, n_tot):
    #
    #     ctx = mp.get_context('forkserver')
    #     queue = ctx.Queue()
    #     envs = [Env() for i in range(args.gpu_workers)]
    #     # new = mp.Event()
    #     jobs = ctx.Queue(n_interval)
    #     done = ctx.Event()
    #
    #     processes = []
    #     for rank in range(args.gpu_workers):
    #         p = ctx.Process(target=self.player, args=(envs[rank], queue, jobs, done, rank))
    #         p.start()
    #         processes.append(p)
    #
    #     try:
    #
    #         results = {'scores': [], 'epoch': []}
    #
    #         for epoch in range(0, n_tot, n_interval):
    #
    #             params = self.model.state_dict()
    #             [jobs.put(params) for i in range(n_interval)]
    #
    #             for i in tqdm(range(n_interval)):
    #                 score = queue.get()
    #                 results['scores'].append(score)
    #                 results['epoch'].append(epoch + i)
    #
    #             yield results
    #             results = {key: [] for key in results}
    #
    #         raise StopIteration
    #
    #     finally:
    #
    #         done.set()
    #         for p in processes:
    #             p.join()




    # # A working example for new process for each iteration
    # def player(env,  model, queue):
    #
    #     print("I am Here")
    #
    #     env.reset()
    #     while not env.t:
    #
    #         s = Variable(env.s, requires_grad=False)
    #
    #         a = model(s)
    #         a = a.data.cpu().numpy()
    #         a = np.argmax(a)
    #         env.step(a)
    #
    #     queue.put(env.score)

    # # A working example for new process for each iteration pretty slow
    # def play(self, n_interval, n_tot):
    #
    #     queue = mp.Queue()
    #     n = args.cpu_workers
    #     envs = [Env() for i in range(args.cpu_workers)]
    #
    #     for epoch in range(0, n_tot, n):
    #
    #         results = {'scores': [], 'epoch': []}
    #         self.models.cpu().eval()
    #         self.models.share_memory()
    #
    #         processes = []
    #         for rank in range(n):
    #             p = mp.Process(target=player, args=(envs[rank], self.models, queue))
    #             p.start()
    #             processes.append(p)
    #
    #         for i in tqdm(range(n)):
    #             score = queue.get()
    #             results['scores'].append(score)
    #             results['epoch'].append(epoch + i)
    #
    #         for p in processes:
    #             p.join()
    #
    #         self.models.cuda().train()
    #         yield results


# def player_q(env, queue, jobs, done, myid):
#
#     # print("P %d: I am Here" % myid)
#     models = {(0,): DQN, (1,): DQNDueling}
#     Model = models[(args.dueling,)]
#     model = Model(consts.n_actions[args.game])
#     model = model.cuda()
#     model = torch.nn.DataParallel(model)
#     greedy = args.greedy
#     action_space = consts.n_actions[args.game]
#
#     while not done.is_set():
#
#         params = jobs.get()
#         model.load_state_dict(params)
#         # print("P %d: got a job" % myid)
#         env.reset()
#         # print("P %d: passed reset" % myid)
#         softmax = torch.nn.Softmax()
#         choices = np.arange(action_space, dtype=np.int)
#
#         while not env.t:
#
#             s = Variable(env.s, requires_grad=False)
#
#             a = model(s)
#             if greedy:
#                 a = a.data.cpu().numpy()
#                 a = np.argmax(a)
#             else:
#                 a = softmax(a).data.squeeze(0).cpu().numpy()
#                 # print(a)
#                 a = np.random.choice(choices, p=a)
#             env.step(a)
#
#         # print("P %d: finished with score %d" % (myid, env.score))
#         queue.put(env.score)
#
#     env.close()
#
# def player_va(env, queue, jobs, done, myid):
#
#     model = DVAN_ActionOut(consts.n_actions[args.game])
#     model = model.cuda()
#     model = torch.nn.DataParallel(model)
#     greedy = args.greedy
#     action_space = consts.n_actions[args.game]
#
#     while not done.is_set():
#
#         params = jobs.get()
#         model.load_state_dict(params)
#         env.reset()
#         softmax = torch.nn.Softmax()
#         choices = np.arange(action_space, dtype=np.int)
#
#         while not env.t:
#
#             s = Variable(env.s, requires_grad=False)
#
#             v, a = model(s)
#             if greedy:
#                 a = a.data.cpu().numpy()
#                 a = np.argmax(a)
#             else:
#                 a = softmax(a).data.squeeze(0).cpu().numpy()
#                 a = np.random.choice(choices, p=a)
#             env.step(a)
#
#         queue.put(env.score)
#
#     env.close()
#
#
# def player_ava(env, queue, jobs, done, myid):
#
#     model = DVAN_ActionIn(3)
#     model = model.cuda()
#     model.eval()
#     model = torch.nn.DataParallel(model)
#     greedy = args.greedy
#     action_space = consts.action_space
#
#     actions_matrix = actions.unsqueeze(0)
#     actions_matrix = actions_matrix.repeat(1, 1, 1).float()
#
#     while not done.is_set():
#
#         params = jobs.get()
#         model.load_state_dict(params)
#         env.reset()
#         softmax = torch.nn.Softmax()
#         choices = np.arange(action_space, dtype=np.int)
#
#         while not env.t:
#
#             s = Variable(env.s, requires_grad=False)
#
#             v, a = model(s, actions_matrix)
#
#             if greedy:
#                 a = a.data.cpu().numpy()
#                 a = np.argmax(a)
#             else:
#                 a = softmax(a).data.squeeze(0).cpu().numpy()
#                 a = np.random.choice(choices, p=a)
#             env.step(a)
#
#         queue.put(env.score)
#
#     env.close()

    # def episodic_evaluator_va(self):
    #
    #     self.model.eval()
    #     self.target.eval()
    #     results = {'q_diff': [], 'a_agent': [], 'a_player': []}
    #
    #     for n, sample in tqdm(enumerate(self.episodic_loader)):
    #
    #         is_final = sample['is_final']
    #         final_indicator, final_index = is_final.max(0)
    #         final_indicator = final_indicator.numpy()[0]
    #         final_index = final_index.numpy()[0]+1 if final_indicator else self.batch
    #
    #         s = Variable(sample['s'][:final_index].cuda(), requires_grad=False)
    #         a = Variable(sample['a'][:final_index].cuda().unsqueeze(1), requires_grad=False)
    #         f = sample['f'][:final_index].float().cuda()
    #         base = sample['base'][:final_index].float()
    #         r = Variable(sample['r'][:final_index].float().cuda(), requires_grad=False)
    #
    #         value, advantage = self.model(s)
    #
    #         value = value.squeeze(1)
    #
    #         if self.myopic:
    #             v_diff = r.data.cpu() - value.data.cpu()
    #         else:
    #             v_diff = (f.cpu() - base) - value.data.cpu()
    #
    #         a_player = a.squeeze(1).data.cpu()
    #         a_agent = advantage.data.cpu()
    #
    #         # add results
    #         results['q_diff'].append(v_diff)
    #         results['a_agent'].append(a_agent)
    #         results['a_player'].append(a_player)
    #
    #         if final_indicator:
    #             results['q_diff'] = torch.cat(results['q_diff'])
    #             results['a_agent'] = torch.cat(results['a_agent'])
    #             results['a_player'] = torch.cat(results['a_player'])
    #             yield results
    #             self.model.eval()
    #             self.target.eval()
    #             results = {key: [] for key in results}
    #
    # def episodic_evaluator_ava(self):
    #
    #     self.model.eval()
    #     self.target.eval()
    #     results = {'q_diff': [], 'a_agent': [], 'a_player': []}
    #
    #     for n, sample in tqdm(enumerate(self.episodic_loader)):
    #
    #         is_final = sample['is_final']
    #         final_indicator, final_index = is_final.max(0)
    #         final_indicator = final_indicator.numpy()[0]
    #         final_index = final_index.numpy()[0]+1 if final_indicator else self.batch
    #
    #         s = Variable(sample['s'][:final_index].cuda(), requires_grad=False)
    #         a = Variable(sample['a'][:final_index].cuda().unsqueeze(1), requires_grad=False)
    #         f = sample['f'][:final_index].float().cuda()
    #         base = sample['base'][:final_index].float()
    #         r = Variable(sample['r'][:final_index].float().cuda(), requires_grad=False)
    #         a_index = Variable(sample['a_index'][:final_index].cuda().unsqueeze(1), requires_grad=False)
    #
    #         value, advantage = self.model(s)
    #         value = value.squeeze(1)
    #
    #         value, advantage = self.model(s, self.actions_matrix)
    #
    #         if self.myopic:
    #             v_diff = r.data.cpu() - value.data.cpu()
    #         else:
    #             v_diff = (f.cpu() - base) - value.data.cpu()
    #
    #         a_player = a.squeeze(1).data.cpu()
    #         a_agent = advantage.data.cpu()
    #
    #         # add results
    #         results['q_diff'].append(v_diff)
    #         results['a_agent'].append(a_agent)
    #         results['a_player'].append(a_player)
    #
    #         if final_indicator:
    #             results['q_diff'] = torch.cat(results['q_diff'])
    #             results['a_agent'] = torch.cat(results['a_agent'])
    #             results['a_player'] = torch.cat(results['a_player'])
    #             yield results
    #             self.model.eval()
    #             self.target.eval()
    #             results = {key: [] for key in results}