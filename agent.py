import torch
import torch.utils.data
import numpy as np
import os
import pandas as pd
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import cv2
from tqdm import tqdm
import time

from demonstration import DemonstrationsDataset
from config import consts, args
from model import DQN
from environment import Env


def cuda(x):
    if args.cuda:
        return x.cuda()
    else:
        return x

class Agent(object):

    def __init__(self, data, meta):

        # define codel components
        self.dataset = DemonstrationsDataset(data, meta)
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.dataset.meta['train'])
        self.test_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.dataset.meta['test'])

        #
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch, sampler=self.train_sampler,
                                                        batch_sampler=None,
                                                        num_workers=args.cpu_workers, pin_memory=True, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.eval_batch,
                                                       sampler=self.test_sampler, batch_sampler=None,
                                                       num_workers=args.cpu_workers, pin_memory=True, drop_last=False)

        # build the action mask tensor
        n = len(consts.activation2action[args.game])
        self.qval = DQN(n)
        self.target = DQN(n)

        # self.qval.cuda()
        # self.target.cuda()

        cuda(self.qval)
        cuda(self.target)

        if args.cuda:
            self.qval = torch.nn.DataParallel(self.qval)
            self.target = torch.nn.DataParallel(self.target)

        # for param in self.qval.parameters():
        #     print(type(param.data), param.size())

        self.model_path = os.path.join(args.indir, 'data', args.identifier)
        if args.load_model:
            self.load_model()

        self.update_target()

        self.loss_fn = torch.nn.MSELoss(size_average=True)
        # self.optimizer = optim.Adam(self.qval.parameters(), lr=args.lr, eps=args.adam_eps)
        # self.optimizer = torch.optim.RMSprop(self.qval.parameters(), lr=0.00025, alpha=0.99, eps=0.01, weight_decay=0, momentum=0.1, centered=True)
        self.optimizer = torch.optim.SGD(self.qval.parameters(), lr=1e-4)


        tensorboard_path = os.path.join(args.indir, 'tensorboard', args.identifier)
        if not os.path.isdir(tensorboard_path):
            os.mkdir(tensorboard_path)
        self.writer = SummaryWriter(log_dir=tensorboard_path, comment=args.identifier)

    def train(self):
        pass

    def lfd(self):
        self.qval.train()
        self.target.eval()

        evaluation = self.evaluate()

        loss_vec = []
        epoch = 0
        while True:
            print("Training epoch number %d" % (epoch+1))
            for i, sample in tqdm(enumerate(self.train_loader)):

                # S should require_grad???????
                s = Variable(sample['s'].cuda(), requires_grad=False)
                s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
                a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
                r = Variable(sample['r'].float().cuda(), requires_grad=False)
                t = Variable(sample['t'].float().cuda(), requires_grad=False)

                q = self.qval(s)
                q_a = q.gather(1, a)[:, 0]

                q_target = self.target(s_tag)
                max_q_target = Variable(q_target.max(1)[0].data, requires_grad=False)

                # self.qval.zero_grad()
                loss = self.loss_fn(q_a, r + args.discount * (max_q_target * (1 - t)))

                self.optimizer.zero_grad()
                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_vec.append(loss.data.mean())

                if i % args.update_interval == 0:
                    self.update_target()
                    if i % args.save_interval == 0:
                        self.save_model()

                if i % args.evaluation_interval == 0:
                    n = i + epoch * self.dataset.meta['n_states']

                    if n == 0:
                        state = Variable(sample['s'].cuda(), requires_grad=False)
                        self.writer.add_graph(self.qval, self.qval(state))

                    x = np.array(loss_vec)
                    avg_train_loss = pd.rolling_mean(x, args.evaluation_interval)[-1]
                    avg_test_loss = next(evaluation)
                    print("Evaluation @ T=%d | Avg. Test Loss: %g |  Avg. Train Loss: %g "
                          % (i, avg_test_loss, avg_train_loss))

                    print(" ")
                    print("Layers statistics:")
                    print(" ")
                    print("conv1:")
                    print("weights norm: %s" % str(self.qval.module.conv1.weight.data.norm()))
                    print("bias norm: %s" % str(self.qval.module.conv1.bias.data.norm()))
                    print("weights-grad norm: %s" % str(self.qval.module.conv1.weight.grad.data.norm()))
                    print("bias-grad norm: %s" % str(self.qval.module.conv1.bias.grad.data.norm()))
                    print(" ")
                    print("conv2:")
                    print("weights norm: %s" % str(self.qval.module.conv2.weight.data.norm()))
                    print("bias norm: %s" % str(self.qval.module.conv2.bias.data.norm()))
                    print("weights-grad norm: %s" % str(self.qval.module.conv2.weight.grad.data.norm()))
                    print("bias-grad norm: %s" % str(self.qval.module.conv2.bias.grad.data.norm()))
                    print(" ")
                    print("conv3:")
                    print("weights norm: %s" % str(self.qval.module.conv3.weight.data.norm()))
                    print("bias norm: %s" % str(self.qval.module.conv3.bias.data.norm()))
                    print("weights-grad norm: %s" % str(self.qval.module.conv3.weight.grad.data.norm()))
                    print("bias-grad norm: %s" % str(self.qval.module.conv3.bias.grad.data.norm()))
                    print(" ")
                    print("fc4:")
                    print("fc4 weights norm: %s" % str(self.qval.module.fc4.weight.data.norm()))
                    print("fc4 bias norm: %s" % str(self.qval.module.fc4.bias.data.norm()))
                    print("fc4 weights-grad norm: %s" % str(self.qval.module.fc4.weight.grad.data.norm()))
                    print("fc4 bias-grad norm: %s" % str(self.qval.module.fc4.bias.grad.data.norm()))
                    print(" ")
                    print("fc5:")
                    print("weights norm: %s" % str(self.qval.module.fc5.weight.data.norm()))
                    print("bias norm: %s" % str(self.qval.module.fc5.bias.data.norm()))
                    print("weights-grad norm: %s" % str(self.qval.module.fc5.weight.grad.data.norm()))
                    print("bias-grad norm: %s" % str(self.qval.module.fc5.bias.grad.data.norm()))
                    print(" ")

                    self.writer.add_scalar('data/avg_test_loss', avg_test_loss, n)
                    self.writer.add_scalar('data/avg_train_loss', avg_train_loss, n)

                    img = s.cpu()[0, 0, :, :].data.numpy()
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = torch.from_numpy(np.rollaxis(img, 2, 0))

                    self.writer.add_image('state', img, n)

                    img = s_tag.cpu()[0, 0, :, :].data.numpy()
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = torch.from_numpy(np.rollaxis(img, 2, 0))

                    self.writer.add_image('next_state', img, n)
                    self.writer.add_scalar('data/action', a.cpu()[0].data.numpy(), n)

                    for name, param in self.qval.named_parameters():
                        self.writer.add_histogram(name, param.clone().cpu().data.numpy(), n)

            epoch += 1

            if epoch * self.dataset.meta['n_states'] >= args.T_max:
                break

    def close(self):
        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(os.path.join(args.indir, 'data', "all_scalars.json"))
        self.writer.close()

    def update_target(self):
        self.target.load_state_dict(self.qval.state_dict())

    def evaluate(self):
        while True:
            for i, sample in enumerate(self.test_loader):

                s = Variable(sample['s'].cuda(), requires_grad=False)
                s_tag = Variable(sample['s_tag'].cuda(), requires_grad=False)
                a = Variable(sample['a'].cuda().unsqueeze(1), requires_grad=False)
                r = Variable(sample['r'].float().cuda(), requires_grad=False)
                t = Variable(sample['t'].float().cuda(), requires_grad=False)

                # s = Variable(cuda(sample['s']), requires_grad=False)
                # s_tag = Variable(cuda(sample['s_tag']), requires_grad=False)
                # a = Variable(cuda(sample['a']).unsqueeze(1), requires_grad=False)
                # r = Variable(cuda(sample['r'].float()), requires_grad=False)
                # t = Variable(cuda(sample['t'].float()), requires_grad=False)

                q = self.qval(s)
                q_a = q.gather(1, a)[:, 0]

                q_target = self.target(s_tag)
                max_q_target = Variable(q_target.max(1)[0].data, requires_grad=False)

                loss = self.loss_fn(q_a, r + args.discount * (max_q_target * (1 - t)))
                yield loss.data.mean()

    def load_model(self):
        self.qval.load_state_dict(torch.load(self.model_path))

    def save_model(self):
        torch.save(self.target.state_dict(), self.model_path)

    def test(self):

        scores = np.zeros(args.test_episodes)
        env = Env()
        self.qval.eval()
        for i in range(args.test_episodes):
            env.reset()
            while not env.t:
                # s = Variable(env.s.cuda(), requires_grad=False)
                # print("new state norm: %g" % env.s.norm())
                s = Variable(cuda(env.s), requires_grad=False)
                a = self.qval(s)
                a = a.data.cpu().numpy()
                a = np.argmax(a)
                # print("Action chosen: %s" % consts.action_meanings[a])
                # if "FIRE" in consts.action_meanings[a]:
                #     a = 1
                # for j in range(args.skip):
                # Skip is implemented in the environment
                env.step(a)
                if args.render:
                    env.render()
                    time.sleep(consts.frame_rate * args.skip)
            print("Game %d ended with score: %d" % (i, env.score))
            scores[i] = env.score

        print("Test Statistics:")
        print("mean: %g" % scores.mean())
        print("std: %g" % scores.std())

        if args.render:
            env.render(close=True)
        env.close()