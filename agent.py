import torch
import torch.utils.data
import torch.utils.data.sampler
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from config import consts, args
from model import DQN
from environment import Env
from memory import DemonstrationMemory, DemonstrationBatchSampler


def player(env, queue, jobs, done, myid):

    # print("P %d: I am Here" % myid)
    model = DQN(consts.n_actions[args.game])
    model.cuda()
    model = torch.nn.DataParallel(model)

    while not done.is_set():

        params = jobs.get()
        model.load_state_dict(params)
        # print("P %d: got a job" % myid)
        env.reset()
        # print("P %d: passed reset" % myid)

        while not env.t:

            s = Variable(env.s, requires_grad=False)

            a = model(s)
            a = a.data.cpu().numpy()
            a = np.argmax(a)
            env.step(a)

        # print("P %d: finished with score %d" % (myid, env.score))
        queue.put(env.score)

    env.close()


class Agent(object):

    def __init__(self):
        self.model = None
        self.optimizer = None

    def set_optimizer(self, params):

        if args.optimizer is "sgd":
            self.optimizer = torch.optim.SGD(params, lr=args.lr,)

        elif args.optimizer is "rmsprop":
            self.optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99, eps=0.01, weight_decay=0, momentum=0.1, centered=True)

        elif args.optimizer is "adam":
            self.optimizer = torch.optim.Adam(params, lr=args.lr, eps=args.adam_eps)

    def save_checkpoint(self, path, aux=None):

        state = {'state_dict': self.model.state_dict(),
                 'optimizer_dict': self.optimizer.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

        return state['aux']

    def train(self, n_interval, n_tot):
        raise NotImplementedError

    def evaluate(self, n_interval, n_tot):
        raise NotImplementedError

    def play(self, n_interval, n_tot):
        raise NotImplementedError

    def resume(self, model_path, optimizer_path):
        raise NotImplementedError


class LfdAgent(Agent):

    def __init__(self):

        super(Agent, self).__init__()

        #
        train_dataset = DemonstrationMemory("train")
        test_dataset = DemonstrationMemory("test")

        train_sampler = DemonstrationBatchSampler(train_dataset, args.batch, False)
        test_sampler = DemonstrationBatchSampler(test_dataset, args.batch, False)


        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                                        num_workers=args.cpu_workers, pin_memory=True, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler,
                                                       num_workers=args.cpu_workers, pin_memory=True, drop_last=False)

        # values of the specific agent
        self.models = DQN(consts.n_actions[args.game])
        self.targets = DQN(consts.n_actions[args.game])
        self.modelcpu = DQN(consts.n_actions[args.game])

        if args.cuda:
            self.models.cuda()
            self.targets.cuda()
            self.model = torch.nn.DataParallel(self.models)
            self.target = torch.nn.DataParallel(self.targets)
        else:
            self.model = self.models
            self.target = self.targets

        self.loss_fn = torch.nn.MSELoss(size_average=True)
        self.set_optimizer(self.model.parameters())

    def resume(self, model_path, optimizer_path):

        aux = self.load_checkpoint(model_path)
        self.update_target()
        return aux

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def train(self, n_interval, n_tot):

        self.model.train()
        self.target.eval()
        results = {'loss': [], 'n': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)

            q = self.model(s)
            q_a = q.gather(1, a)[:, 0]

            q_target = self.target(s_tag)
            max_q_target = Variable(q_target.max(1)[0].data, requires_grad=False)

            loss = self.loss_fn(q_a, r + args.discount * (max_q_target * (1 - t)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)

            if n % args.update_interval == 0:
                self.update_target()

            if not (n+1) % n_interval:
                yield results
                self.model.train()
                self.target.eval()
                results = {'loss': [], 'epoch': [], 'n': []}

    def evaluate(self, n_interval, n_tot):

        self.model.eval()
        self.target.eval()

        results = {'loss': [], 'n': []}

        for n, sample in enumerate(self.test_loader):

            s = Variable(sample['s'].cuda(), requires_grad=False)
            s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
            a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
            r = Variable(sample['r'].float().cuda(), requires_grad=False)
            t = Variable(sample['t'].float().cuda(), requires_grad=False)

            q = self.model(s)
            q_a = q.gather(1, a)[:, 0]

            q_target = self.target(s_tag)
            max_q_target = Variable(q_target.max(1)[0].data, requires_grad=False)

            loss = self.loss_fn(q_a, r + args.discount * (max_q_target * (1 - t)))

            # add results
            results['loss'].append(loss.data.cpu().numpy()[0])
            results['n'].append(n)

            if not (n+1) % n_interval:
                results['s'] = s.data.cpu()
                results['s_tag'] = s_tag.data.cpu()
                results['q'] = q.data.cpu()
                results['r'] = r.data.cpu()
                yield results
                self.model.eval()
                self.target.eval()
                results = {'loss': [], 'epoch': [], 'n': []}


    def play(self, n_interval, n_tot):

        ctx = mp.get_context('forkserver')
        queue = ctx.Queue()
        envs = [Env() for i in range(args.gpu_workers)]
        # new = mp.Event()
        jobs = ctx.Queue(n_interval)
        done = ctx.Event()

        # self.modelcpu.share_memory()

        processes = []
        for rank in range(args.gpu_workers):
            p = ctx.Process(target=player, args=(envs[rank], queue, jobs, done, rank))
            p.start()
            processes.append(p)

        for epoch in range(0, n_tot, n_interval):

            results = {'scores': [], 'epoch': []}
            # self.modelcpu.load_state_dict(self.models.cpu().state_dict())
            # self.models.cuda()

            params = self.model.state_dict()

            [jobs.put(params) for i in range(n_interval)]

            for i in tqdm(range(n_interval)):
                score = queue.get()
                results['scores'].append(score)
                results['epoch'].append(epoch + i)

            yield results

        done.set()
        for p in processes:
            p.join()




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


