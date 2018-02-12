import csv
import time
import os
import sys
import numpy as np
import torch

from tensorboardX import SummaryWriter
import comet_ml as comet
import torch
from torch.autograd import Variable
import cv2
from tqdm import tqdm
import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from config import consts, args
from agent import LfdAgent
from test_agent import TestAgent
from behavioral_agent import BehavioralAgent
from behavioral_hot_agent import BehavioralHotAgent
from behavioral_dist_agent import BehavioralDistAgent
from cdqn_horizon_agent import ACDQNLSTMAgent
from acdqn_agent import ACDQNAgent
# from detached_agent import DetachedAgent
from partially_detached_agent import DetachedAgent
from logger import logger
from preprocess import convert_screen_to_rgb
from distutils.dir_util import copy_tree


class Experiment(object):

    def __init__(self, load_exp=None):

        # parameters
        self.input_actions = True

        if self.input_actions:
            self.action_space = consts.action_space
        else:
            self.action_space = consts.n_actions[args.game]

        self.action_meanings = consts.action_meanings
        self.one_hots = torch.sparse.torch.eye(self.action_space)
        self.batch = args.batch
        self.l1_loss = args.l1_loss
        self.detached_agent = args.detached_agent

        dirs = os.listdir(args.outdir)

        if args.comet:
            self.comet = comet.Experiment(api_key=consts.api_key)
            self.comet.log_multiple_params(vars(args))

        self.load_model = args.load_last_model or args.load_best_model
        self.load_best = args.load_best_model
        self.load_last = args.load_last_model

        if load_exp is not None:
            self.load_last = True
            self.resume = load_exp
        else:
            self.resume = args.resume

        self.exp_name = ""
        if self.load_model:
            if self.resume >= 0:
                for d in dirs:
                    if "%s_exp_%04d_" % (args.identifier, self.resume) in d:
                        self.exp_name = d
                        self.exp_num = self.resume
                        break
            else:
                raise Exception("Non-existing experiment")

        if not self.exp_name:
            # count similar experiments
            n = sum([1 for d in dirs if "%s_exp" % args.identifier in d])
            self.exp_name = "%s_exp_%04d_%s" % (args.identifier, n, consts.exptime)
            self.load_model = False

            self.exp_num = n

        # init experiment parameters
        self.root = os.path.join(args.outdir, self.exp_name)

        # set dirs
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.scores_dir = os.path.join(self.root, 'scores')
        self.code_dir = os.path.join(self.root, 'code')

        self.checkpoint = os.path.join(self.checkpoints_dir, 'checkpoint')
        self.checkpoint_best = os.path.join(self.checkpoints_dir, 'checkpoint_best')

        if self.load_model:
            logger.info("Resuming existing experiment")
        else:
            logger.info("Creating new experiment")
            os.makedirs(self.root)
            os.makedirs(self.tensorboard_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.results_dir)
            os.makedirs(self.scores_dir)
            os.makedirs(self.code_dir)

            # copy code to dir
            copy_tree(os.path.abspath("."), self.code_dir)

            # write csv file of hyper-parameters
            filename = os.path.join(self.root, "hyperparameters.csv")
            with open(filename, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(self.exp_name)
                for k, v in vars(args).items():
                    spamwriter.writerow([k, str(v)])

        # initialize tensorboard writer
        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=args.identifier)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()

    def visualize(self, model_name):
        agent = self.choose_agent()
        model = getattr(agent, model_name)
        conv1 = model.cnn_conv1[0].weight.data.cpu().numpy()
        n, c, x, y = conv1.shape
        conv1_bw = conv1.transpose(0, 2, 1, 3).reshape(n * x, c * y)

        conv1_3d = conv1[:, :3, :, :]
        conv1_3d = np.expand_dims(conv1_3d, axis=0)
        conv1_3d = conv1_3d.reshape(8, 4, 3, 8, 8)
        conv1_3d = conv1_3d.transpose(0, 3, 1, 4, 2)
        conv1_3d = conv1_3d.reshape(64, 32, 3)

        return {'conv1_bw': conv1_bw,
                'conv1_3d': conv1_3d,
                'conv1': conv1
        }

    def behavioral(self):

        if args.distributional:
            return self.behavioral_distributional()
        elif args.actor_critic:
            return self.behavioral_ac()
        else:
            return self.behavioral_hot()

    def behavioral_ac(self):

        # init time variables
        t_start = time.time()

        if args.detached_agent:
            agent = DetachedAgent()

        elif not args.recurrent:
            agent = ACDQNAgent()
        else:
            agent = ACDQNLSTMAgent()

        # load model
        if self.load_model:
            if self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            n_offset = aux['n']
            # n_offset = 61
        else:
            n_offset = 0

            # save a random init checkpoint
            agent.save_checkpoint(self.checkpoint, {'n': 0})

        best = np.inf

        # define experiment generators
        test = agent.test(args.evaluate_frames, args.n_tot)
        learn = agent.learn(args.checkpoint_interval, args.n_tot)

        human_score = agent.train_dataset.meta['avg_score']

        test_results = next(test)

        init_test_loss = np.mean(test_results['loss_beta'])

        t_last = time.time()

        logger.info("Begin Behavioral Distributional learning experiment")
        logger.info("Game: %s | human score %g | initial loss %g" %
                    (args.game, human_score, init_test_loss))

        for n, train_results in enumerate(learn):

            t_train = time.time() - t_last
            test_results = next(test)
            t_eval = time.time() - t_train - t_last

            simulation_time = time.time() - t_start

            avg_train_loss_beta = np.mean(train_results['loss_beta'])
            avg_test_loss_beta = np.mean(test_results['loss_beta'])

            avg_train_loss_v_beta = np.mean(train_results['loss_v_beta'])
            avg_test_loss_v_beta = np.mean(test_results['loss_v_beta'])

            avg_train_loss_q_beta = np.mean(train_results['loss_q_beta'])
            avg_test_loss_q_beta = np.mean(test_results['loss_q_beta'])

            avg_train_loss_q_pi = np.mean(train_results['loss_q_pi'])
            avg_test_loss_q_pi = np.mean(test_results['loss_q_pi'])

            avg_train_loss_v_pi = np.mean(train_results['loss_v_pi'])
            avg_test_loss_v_pi = np.mean(test_results['loss_v_pi'])

            avg_train_loss_pi = np.mean(train_results['loss_pi'])
            avg_test_loss_pi = np.mean(test_results['loss_pi'])

            avg_act_diff = np.mean(test_results['act_diff'])

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('train_loss/loss_beta', float(avg_train_loss_beta), n + n_offset)
                self.writer.add_scalar('test_loss/loss_beta', float(avg_test_loss_beta), n + n_offset)

                self.writer.add_scalar('train_loss/loss_q_beta', float(avg_train_loss_q_beta), n + n_offset)
                self.writer.add_scalar('test_loss/loss_q_beta', float(avg_test_loss_q_beta), n + n_offset)

                self.writer.add_scalar('train_loss/loss_v_beta', float(avg_train_loss_v_beta), n + n_offset)
                self.writer.add_scalar('test_loss/loss_v_beta', float(avg_test_loss_v_beta), n + n_offset)

                self.writer.add_scalar('train_loss/loss_v_pi', float(avg_train_loss_v_pi), n + n_offset)
                self.writer.add_scalar('test_loss/loss_v_pi', float(avg_test_loss_v_pi), n + n_offset)

                self.writer.add_scalar('train_loss/loss_q_pi', float(avg_train_loss_q_pi), n + n_offset)
                self.writer.add_scalar('test_loss/loss_q_pi', float(avg_test_loss_q_pi), n + n_offset)

                self.writer.add_scalar('train_loss/loss_pi', float(avg_train_loss_pi), n + n_offset)
                self.writer.add_scalar('test_loss/loss_pi', float(avg_test_loss_pi), n + n_offset)

                self.writer.add_scalar('actions/act_diff', float(avg_act_diff), n)

                self.writer.add_histogram("actions/agent", test_results['a_agent'], n, 'doane')
                self.writer.add_histogram("actions/a_player", test_results['a_player'], n, 'doane')

                if not self.detached_agent:
                    for name, param in agent.model.named_parameters():
                        self.writer.add_histogram("model/%s" % name, param.clone().cpu().data.numpy(), n + n_offset, 'fd')
                else:
                    for name, param in agent.beta_net.named_parameters():
                        self.writer.add_histogram("beta_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset, 'fd')
                    for name, param in agent.pi_net.named_parameters():
                        self.writer.add_histogram("pi_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset, 'fd')
                    for name, param in agent.q_net.named_parameters():
                        self.writer.add_histogram("q_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset, 'fd')
                    # for name, param in agent.vb_net.named_parameters():
                    #     self.writer.add_histogram("vb_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset, 'fd')

            # save agent state
            agent.save_checkpoint(self.checkpoint, {'n': n + 1})

            if abs(avg_train_loss_beta) < best:
                best = abs(avg_train_loss_beta)
                agent.save_checkpoint(self.checkpoint_best, {'n': n + 1})

            # init interval timer
            t_interval = time.time() - t_last
            t_last = time.time()

            img = test_results['s'][0, :-1, :, :]

            self.writer.add_image('states/state', img, n)

            if not self.detached_agent:
                logger.info(" ")
                #  log to screen and logger
                logger.info("------------Checkpoint @ Behavioral Experiment: %s------------" % self.exp_name)
                logger.info(
                    "Evaluation @ n=%d, step=%d | Test Qpi Loss: %g | Train Vpi Loss: %g | Train pi Loss: %g | Train beta Loss: %g"
                    % (n + n_offset, train_results['n'][-1], avg_test_loss_q_pi, avg_test_loss_v_pi, avg_test_loss_pi,
                       avg_test_loss_beta))
                logger.info("Total Simulation time: %g| Interval time: %g| Train: %g |  Test: %g"
                            % (simulation_time, t_interval, t_train, t_eval))

                logger.info(" ")
                logger.info("Layers statistics:")
                for name, module in list(agent.model.named_modules()):

                    if not hasattr(module, 'weight'):
                        continue
                    logger.info(" ")
                    logger.info("%s:" % name)
                    logger.info("weights norm: %s" % str(module.weight.data.norm()))

                    if module.weight.grad is not None:
                        logger.info("weights-grad norm: %s" % str(module.weight.grad.data.norm()))

                    if not hasattr(module, 'bias'):
                        continue

                    logger.info("bias norm: %s" % str(module.bias.data.norm()))

                    if module.bias.grad is not None:
                        logger.info("bias-grad norm: %s" % str(module.bias.grad.data.norm()))

            if not args.recurrent:
                self.print_actions_statistics(test_results['a_agent'], test_results['a_player'], n)

        return agent


    def behavioral_distributional(self):

        # init time variables
        t_start = time.time()

        agent = BehavioralDistAgent()

        # load model
        if self.load_model:
            if self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            n_offset = aux['n']
            # n_offset = 61
        else:
            n_offset = 0

        best = np.inf

        # define experiment generators
        test = agent.test(args.evaluate_frames, args.n_tot)
        learn = agent.learn(args.checkpoint_interval, args.n_tot)

        human_score = agent.train_dataset.meta['avg_score']

        test_results = next(test)

        init_test_loss = np.mean(test_results['loss_vs'])

        t_last = time.time()

        logger.info("Begin Behavioral Distributional learning experiment")
        logger.info("Game: %s | human score %g | initial loss %g" %
                    (args.game, human_score, init_test_loss))

        for n, train_results in enumerate(learn):

            t_train = time.time() - t_last
            test_results = next(test)
            t_eval = time.time() - t_train - t_last

            simulation_time = time.time() - t_start

            avg_train_loss_qs = np.mean(train_results['loss_qs'])
            avg_test_loss_qs = np.mean(test_results['loss_qs'])

            avg_train_loss_vs = np.mean(train_results['loss_vs'])
            avg_test_loss_vs = np.mean(test_results['loss_vs'])

            avg_train_loss_b = np.mean(train_results['loss_b'])
            avg_test_loss_b = np.mean(test_results['loss_b'])

            avg_train_loss_vl = np.mean(train_results['loss_vl'])
            avg_test_loss_vl = np.mean(test_results['loss_vl'])

            avg_train_loss_ql = np.mean(train_results['loss_ql'])
            avg_test_loss_ql = np.mean(test_results['loss_ql'])

            avg_train_loss_pi_s = np.mean(train_results['loss_pi_s'])
            avg_test_loss_pi_s = np.mean(test_results['loss_pi_s'])

            avg_train_loss_pi_l = np.mean(train_results['loss_pi_l'])
            avg_test_loss_pi_l = np.mean(test_results['loss_pi_l'])

            avg_train_loss_pi_s_tau = np.mean(train_results['loss_pi_s_tau'])
            avg_test_loss_pi_s_tau = np.mean(test_results['loss_pi_s_tau'])

            avg_train_loss_pi_l_tau = np.mean(train_results['loss_pi_l_tau'])
            avg_test_loss_pi_l_tau = np.mean(test_results['loss_pi_l_tau'])

            avg_act_diff = np.mean(test_results['act_diff'])

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('train_loss/loss_qs', float(avg_train_loss_qs), n + n_offset)
                self.writer.add_scalar('test_loss/loss_qs', float(avg_test_loss_qs), n + n_offset)

                self.writer.add_scalar('train_loss/loss_vs', float(avg_train_loss_vs), n + n_offset)
                self.writer.add_scalar('test_loss/loss_vs', float(avg_test_loss_vs), n + n_offset)

                self.writer.add_scalar('train_loss/loss_b', float(avg_train_loss_b), n + n_offset)
                self.writer.add_scalar('test_loss/loss_b', float(avg_test_loss_b), n + n_offset)

                self.writer.add_scalar('train_loss/loss_ql', float(avg_train_loss_ql), n + n_offset)
                self.writer.add_scalar('test_loss/loss_ql', float(avg_test_loss_ql), n + n_offset)

                self.writer.add_scalar('train_loss/loss_vl', float(avg_train_loss_vl), n + n_offset)
                self.writer.add_scalar('test_loss/loss_vl', float(avg_test_loss_vl), n + n_offset)

                self.writer.add_scalar('train_loss/loss_pi_s', float(avg_train_loss_pi_s), n + n_offset)
                self.writer.add_scalar('test_loss/loss_pi_s', float(avg_test_loss_pi_s), n + n_offset)

                self.writer.add_scalar('train_loss/loss_pi_l', float(avg_train_loss_pi_l), n + n_offset)
                self.writer.add_scalar('test_loss/loss_pi_l', float(avg_test_loss_pi_l), n + n_offset)

                self.writer.add_scalar('train_loss/loss_pi_s_tau', float(avg_train_loss_pi_s_tau), n + n_offset)
                self.writer.add_scalar('test_loss/loss_pi_s_tau', float(avg_test_loss_pi_s_tau), n + n_offset)

                self.writer.add_scalar('train_loss/loss_pi_l_tau', float(avg_train_loss_pi_l_tau), n + n_offset)
                self.writer.add_scalar('test_loss/loss_pi_l_tau', float(avg_test_loss_pi_l_tau), n + n_offset)

                self.writer.add_scalar('actions/act_diff', float(avg_act_diff), n)

                self.writer.add_histogram("actions/agent", test_results['a_agent'], n, 'doane')
                self.writer.add_histogram("actions/a_player", test_results['a_player'], n, 'doane')


                for name, param in agent.model.named_parameters():
                    self.writer.add_histogram("model/%s" % name, param.clone().cpu().data.numpy(), n + n_offset, 'fd')

            # save agent state
            agent.save_checkpoint(self.checkpoint, {'n': n + 1})

            if abs(avg_test_loss_vs) < best:
                best = abs(avg_test_loss_vs)
                agent.save_checkpoint(self.checkpoint_best, {'n': n + 1})

            # init interval timer
            t_interval = time.time() - t_last
            t_last = time.time()

            img = test_results['s'][0, :-1, :, :]

            self.writer.add_image('states/state', img, n)

            logger.info(" ")
            #  log to screen and logger
            logger.info("------------Checkpoint @ Behavioral Experiment: %s------------" % self.exp_name)
            logger.info(
                "Evaluation @ n=%d, step=%d | Test Qs Loss: %g | Train Qs Loss: %g | Train Vs Loss: %g | Train beta Loss: %g"
                % (n + n_offset, train_results['n'][-1], avg_test_loss_qs, avg_train_loss_qs, avg_train_loss_vs,
                   avg_train_loss_b))
            logger.info("Total Simulation time: %g| Interval time: %g| Train: %g |  Test: %g"
                        % (simulation_time, t_interval, t_train, t_eval))

            logger.info(" ")
            logger.info("Layers statistics:")
            for name, module in list(agent.model.named_modules()):

                if not hasattr(module, 'weight'):
                    continue
                logger.info(" ")
                logger.info("%s:" % name)
                logger.info("weights norm: %s" % str(module.weight.data.norm()))

                if module.weight.grad is not None:
                    logger.info("weights-grad norm: %s" % str(module.weight.grad.data.norm()))

                if not hasattr(module, 'bias'):
                    continue

                logger.info("bias norm: %s" % str(module.bias.data.norm()))

                if module.bias.grad is not None:
                    logger.info("bias-grad norm: %s" % str(module.bias.grad.data.norm()))

            self.print_actions_statistics(test_results['a_agent'], test_results['a_player'], n)

        return agent.model.state_dict()

    def behavioral_hot(self):

        # init time variables
        t_start = time.time()

        if args.hot:
            agent = BehavioralHotAgent()
        else:
            agent = BehavioralAgent()

        # load model
        if self.load_model:
            if self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            n_offset = aux['n']
            # n_offset = 61
        else:
            n_offset = 0

        best = np.inf

        # define experiment generators
        test = agent.test(args.evaluate_frames, args.n_tot)
        episodic = agent.episodic_evaluator()
        learn = agent.learn(args.checkpoint_interval, args.n_tot)

        human_score = agent.train_dataset.meta['avg_score']

        episodic_results = next(episodic)
        test_results = next(test)

        init_test_loss = np.mean(test_results['loss_v'])

        t_last = time.time()

        logger.info("Begin Behavioral learning experiment")
        logger.info("Game: %s | human score %g | initial loss %g" %
                    (args.game, human_score, init_test_loss))

        for n, train_results in enumerate(learn):

            t_train = time.time() - t_last
            test_results = next(test)
            t_eval = time.time() - t_train - t_last
            episodic_results = next(episodic)
            t_episodic = time.time() - t_eval - t_train - t_last

            simulation_time = time.time() - t_start

            avg_train_loss_q = np.mean(train_results['loss_q'])
            avg_test_loss_q = np.mean(test_results['loss_q'])

            avg_train_loss_v = np.mean(train_results['loss_v'])
            avg_train_loss_b = np.mean(train_results['loss_b'])
            avg_test_loss_v = np.mean(test_results['loss_v'])
            avg_test_loss_b = np.mean(test_results['loss_b'])

            avg_train_loss_r = np.mean(train_results['loss_r'])
            avg_train_loss_p = np.mean(train_results['loss_p'])
            avg_test_loss_r = np.mean(test_results['loss_r'])
            avg_test_loss_p = np.mean(test_results['loss_p'])

            avg_act_diff = np.mean(test_results['act_diff'])

            # std_train_loss = np.std(train_results['loss'])
            # std_test_loss = np.std(test_results['loss'])
            # std_train_loss_v = np.std(train_results['loss_v'])
            # std_train_loss_b = np.std(train_results['loss_b'])
            # std_test_loss_v = np.std(test_results['loss_v'])
            # std_test_loss_b = np.std(test_results['loss_b'])

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('train_loss/loss_q', float(avg_train_loss_q), n + n_offset)
                self.writer.add_scalar('test_loss/loss_q', float(avg_test_loss_q), n + n_offset)

                self.writer.add_scalar('train_loss/loss_v', float(avg_train_loss_v), n + n_offset)
                self.writer.add_scalar('test_loss/loss_v', float(avg_test_loss_v), n + n_offset)

                self.writer.add_scalar('train_loss/loss_b', float(avg_train_loss_b), n + n_offset)
                self.writer.add_scalar('test_loss/loss_b', float(avg_test_loss_b), n + n_offset)

                self.writer.add_scalar('train_loss/loss_p', float(avg_train_loss_p), n + n_offset)
                self.writer.add_scalar('test_loss/loss_p', float(avg_test_loss_p), n + n_offset)

                self.writer.add_scalar('train_loss/loss_r', float(avg_train_loss_r), n + n_offset)
                self.writer.add_scalar('test_loss/loss_r', float(avg_test_loss_r), n + n_offset)

                self.writer.add_scalar('actions/act_diff', float(avg_act_diff), n)

                self.writer.add_histogram("actions/agent", test_results['a_agent'], n, 'doane')
                self.writer.add_histogram("actions/a_player", test_results['a_player'], n, 'doane')

                # self.writer.add_scalar('loss/std_train_loss', float(std_train_loss), n + n_offset)
                # self.writer.add_scalar('loss/std_test_loss', float(std_test_loss), n + n_offset)
                #
                # self.writer.add_scalar('loss/std_train_loss_v', float(std_train_loss_v), n + n_offset)
                # self.writer.add_scalar('loss/std_test_loss_v', float(std_test_loss_v), n + n_offset)
                #
                # self.writer.add_scalar('loss/std_train_loss_b', float(std_train_loss_b), n + n_offset)
                # self.writer.add_scalar('loss/std_test_loss_b', float(std_test_loss_b), n + n_offset)

                for name, param in agent.model.named_parameters():
                    # print(name)
                    # print(param.data.cpu().numpy().max())
                    # print(param.data.cpu().numpy().min())
                    self.writer.add_histogram("model/%s" % name, param.clone().cpu().data.numpy(), n + n_offset, 'fd')

            # save agent state
            agent.save_checkpoint(self.checkpoint, {'n': n + 1})

            if abs(avg_test_loss_v) < best:
                best = abs(avg_test_loss_v)
                agent.save_checkpoint(self.checkpoint_best, {'n': n + 1})

            # init interval timer
            t_interval = time.time() - t_last
            t_last = time.time()

            img = test_results['s'][0, :-1, :, :]

            self.writer.add_image('states/state', img, n)

            logger.info(" ")
            #  log to screen and logger
            logger.info("------------Checkpoint @ Behavioral Experiment: %s------------" % self.exp_name)
            logger.info(
                "Evaluation @ n=%d, step=%d | Test Q Loss: %g | Train Q Loss: %g | Train V Loss: %g | Train beta Loss: %g"
                % (n + n_offset, train_results['n'][-1], avg_test_loss_q, avg_train_loss_q, avg_train_loss_v,
                   avg_train_loss_b))
            logger.info("Total Simulation time: %g| Interval time: %g| Train: %g |  Test: %g"
                        % (simulation_time, t_interval, t_train, t_eval))

            logger.info(" ")
            logger.info("Layers statistics:")
            for name, module in list(agent.model.named_modules()):

                if not hasattr(module, 'weight'):
                    continue
                logger.info(" ")
                logger.info("%s:" % name.split(".")[-1])
                logger.info("weights norm: %s" % str(module.weight.data.norm()))

                if module.weight.grad is not None:
                    logger.info("weights-grad norm: %s" % str(module.weight.grad.data.norm()))

                if not hasattr(module, 'bias'):
                    continue

                logger.info("bias norm: %s" % str(module.bias.data.norm()))

                if module.bias.grad is not None:
                    logger.info("bias-grad norm: %s" % str(module.bias.grad.data.norm()))
                    # except:
                    #     print("XXX")

            self.print_actions_statistics(test_results['a_agent'], test_results['a_player'], n)

        return agent.model.state_dict()

    def print_actions_statistics(self, a_agent, a_player, n):

        # print action meanings
        logger.info("Actions statistics")

        line = ''
        line += "|\tActions Names\t"
        for a in self.action_meanings:
            line += "|%s%s" % (a[:11], ' '*(11 - len(a[:11])))
        line += "|"
        logger.info(line)

        n_actions = len(a_agent)
        applied_player_actions = (np.bincount(np.concatenate((a_player, np.arange(self.action_space)))) - 1) / n_actions
        applied_agent_actions = (np.bincount(np.concatenate((a_agent, np.arange(self.action_space)))) - 1) / n_actions

        line = ''
        line += "|\tPlayer actions\t"
        for a in applied_player_actions:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

        line = ''
        line += "|\tAgent actions\t"
        for a in applied_agent_actions:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

        match_precentage_by_action = []
        error_precentage_by_action = []
        for a in range(len(self.action_meanings)):

            n_action_player = (a_player == a).sum()
            if n_action_player:
                match_precentage_by_action.append((a_agent[a_player == a] == a).sum() / n_action_player)
            else:
                match_precentage_by_action.append(-0.01)

            n_not_action_player = (a_player != a).sum()
            if n_not_action_player:
                error_precentage_by_action.append((a_agent[a_player != a] == a).sum() / n_not_action_player)
            else:
                error_precentage_by_action.append(-0.01)

        line = ''
        line += "|\tMatch by action\t"
        for a in match_precentage_by_action:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

        line = ''
        line += "|\tError by action\t"
        for a in error_precentage_by_action:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)


    def play_behavioral_render(self, params=None):

        agent = self.choose_agent()

        # load model
        if params is not None:
            agent.resume(params)
        elif self.load_last:
            agent.resume(self.checkpoint)
        elif self.load_best:
            agent.resume(self.checkpoint_best)

        player = agent.play_episode(args.test_episodes)

        for i, step in enumerate(player):
            # print("i=%d" % i)
            yield step


    def choose_agent(self):

        if args.detached_agent:
            agent = DetachedAgent(load_dataset=False)

        elif args.distributional:
            agent = BehavioralDistAgent(load_dataset=False)

        elif args.actor_critic:
            agent = ACDQNAgent(load_dataset=False)

        else:
            if args.hot:
                agent = BehavioralHotAgent(load_dataset=False)
            else:
                agent = BehavioralAgent(load_dataset=False)

        return agent

    def play_behavioral(self, params=None):

        agent = self.choose_agent()

        # load model
        if params is not None:
            agent.resume(params)
        elif self.load_last:
            agent.resume(self.checkpoint)
        elif self.load_best:
            agent.resume(self.checkpoint_best)

        # player = agent.play(args.test_episodes)
        player = agent.play_episode(args.test_episodes)

        for i, step in enumerate(player):
            score = step['score']
            print("episode %d | score is %d" % (i, score))

    def play(self, params=None):

        agent = self.choose_agent()
        n = 0

        while n * args.checkpoint_interval < (args.n_tot - 2 * 4096):

            # load model
            try:
                if params is not None:
                    aux = agent.resume(params)
                elif self.load_last:
                    aux = agent.resume(self.checkpoint)
                elif self.load_best:
                    aux = agent.resume(self.checkpoint_best)
                else:
                    raise NotImplementedError
            except: # when reading and writing collide
                time.sleep(2)
                if params is not None:
                    aux = agent.resume(params)
                elif self.load_last:
                    aux = agent.resume(self.checkpoint)
                elif self.load_best:
                    aux = agent.resume(self.checkpoint_best)
                else:
                    raise NotImplementedError

            results = {'n': aux['n'], 'beta':{'score':[], 'frames':[]},
                       'q_b': {'score': [], 'frames': []},
                       'pi': {'score': [], 'frames': []},
                       'q_pi': {'score': [], 'frames': []}}

            player_types = ['beta', 'q_b', 'pi', 'q_pi']

            for action_offset in [1, round((args.action_offset + 1) / 2.), args.action_offset]:

                for type in player_types:
                    player = agent.play(args.test_episodes, action_offset, type)
                    results[type]['score'].append([])
                    results[type]['frames'].append([])

                    for i, step in tqdm(enumerate(player)):
                        results[type]['score'][-1].append(step['score'])
                        results[type]['frames'][-1].append(step['frames'] * args.skip)

                n = results['n']

                logger.info("Episode %d | action offset %d" % (n, action_offset))
                for type in player_types:
                    score = np.array(results[type]['score'][-1])
                    frames = np.array(results[type]['frames'][-1])
                    logger.info("Player: %s\t|avg score: %d\t|avg_frames: %d\t|best score: %d\t| worst score: %d\t| " % (type, round(score.mean()), round(frames.mean()), score.max(), score.min()))

            filename = os.path.join(self.scores_dir, "%d" % n)
            np.savez(filename, results=results)

    def lfd(self):

        # init time variables
        t_start = time.time()

        agent = LfdAgent()

        # load model
        if self.load_model:
            if self.load_last:
                agent.resume(self.checkpoint)
            elif self.load_best:
                agent.resume(self.checkpoint_best)

        best = np.inf

        # define experiment generators
        test = agent.test(args.evaluate_frames, args.n_tot)
        episodic = agent.episodic_evaluator()
        play = agent.play(args.test_episodes, args.n_tot)
        learn = agent.learn(args.checkpoint_interval, args.n_tot)

        human_score = agent.train_dataset.meta['avg_score']
        # calculate random score

        play_results = next(play)
        random_score = np.mean(play_results['scores'])
        # random_score = 0
        # save graph to tensorboard
        episodic_results = next(episodic)
        test_results = next(test)

        # s = Variable(test_results['s'], requires_grad=False)
        # self.writer.add_graph(agent.model[0], agent.model(s))
        init_test_loss = np.mean(test_results['loss'])

        t_last = time.time()

        logger.info("Episodes list by execution:")
        logger.info(agent.meta['episodes'])
        logger.info(" ")
        logger.info("Begin LfD experiment")
        logger.info("Game: %s | human score %g | random score %g | initial q loss %g" %
                    (args.game, human_score, random_score, init_test_loss))

        # set a fix random score (from the atari grand challenge paper):
        random_score = consts.random_score[args.game]

        for n, train_results in enumerate(learn):

            t_train = time.time() - t_last
            test_results = next(test)
            t_eval = time.time() - t_train - t_last
            play_results = next(play)
            t_play = time.time() - t_eval - t_train - t_last
            episodic_results = next(episodic)
            t_episodic = time.time() - t_eval - t_train - t_last - t_play

            simulation_time = time.time() - t_start

            avg_train_loss = np.mean(train_results['loss'])
            avg_test_loss = np.mean(test_results['loss'])

            if self.l1_loss:
                avg_train_score_loss = avg_train_loss * human_score
                avg_test_score_loss = avg_test_loss * human_score
            else:
                avg_train_score_loss = np.sqrt(avg_train_loss) * human_score
                avg_test_score_loss = np.sqrt(avg_test_loss) * human_score

            std_train_loss = np.std(train_results['loss'])
            std_test_loss = np.std(test_results['loss'])

            avg_act_diff = np.mean(test_results['act_diff'])
            std_act_diff = np.std(test_results['act_diff'])

            avg_test_q_diff = np.mean(test_results['q_diff'].numpy())
            std_test_q_diff = np.std(test_results['q_diff'].numpy())

            # see page 6 at paper: Deep Reinforcement Learning with Double Q-Learning

            normalized_score = (np.array(play_results['scores']) - random_score) / (human_score - random_score)

            avg_score = np.mean(normalized_score)
            std_score = np.std(normalized_score)

            # Log to Comet.ml
            if args.comet:
                self.comet.log_step(n)
                self.comet.log_loss(avg_test_loss)
                self.comet.log_accuracy(avg_score)

                self.comet.log_metric("test_loss", avg_test_loss)
                self.comet.log_metric("train_loss", avg_train_loss)
                self.comet.log_metric("avg_score", avg_score)
                self.comet.log_metric("std_score", std_score)
                self.comet.log_metric("train_time", t_train)
                self.comet.log_metric("play_time", t_play)
                self.comet.log_metric("simulation_time", simulation_time)

            # log to tensorboard

            self.writer.add_scalar('loss/avg_train_loss', float(avg_train_loss), n)
            self.writer.add_scalar('loss/avg_test_loss', float(avg_test_loss), n)

            self.writer.add_scalar('loss/avg_train_score_loss', float(avg_train_score_loss), n)
            self.writer.add_scalar('loss/avg_test_score_loss', float(avg_test_score_loss), n)

            self.writer.add_scalar('loss/std_train_loss', float(std_train_loss), n)
            self.writer.add_scalar('loss/std_test_loss', float(std_test_loss), n)
            self.writer.add_scalar('loss/avg_test_q_diff', float(avg_test_q_diff), n)
            self.writer.add_scalar('loss/std_test_q_diff', float(std_test_q_diff), n)
            self.writer.add_scalar('score/avg_score', float(avg_score), n)
            self.writer.add_scalar('score/std_score', float(std_score), n)
            self.writer.add_scalar('actions/avg_act_diff', float(avg_act_diff), n)
            self.writer.add_scalar('actions/std_act_diff', float(std_act_diff), n)

            self.writer.add_scalar('meta/train_time', float(t_train), n)
            self.writer.add_scalar('meta/play_time', float(t_play), n)
            self.writer.add_scalar('meta/simulation_time', float(simulation_time), n)
            self.writer.add_scalar('meta/train_n_steps', float(train_results['n_steps']), n)
            self.writer.add_scalar('meta/test_n_steps', float(test_results['n_steps']), n)
            # change to something more generic
            self.writer.add_scalar('meta/optimizer_lr', float(agent.optimizer.param_groups[0]['lr']), n)

            img = test_results['s'][0, :-1, :, :]
            # img = convert_screen_to_rgb(img)

            self.writer.add_image('states/state', img, n)

            img = test_results['s_tag'][0, :-1, :, :]
            # img = convert_screen_to_rgb(img)

            self.writer.add_image('states/next_state', img, n)

            for name, param in agent.model.named_parameters():
                self.writer.add_histogram("model/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')

            for i, q in enumerate(episodic_results['q_diff']):
                self.writer.add_scalar("episodes/q_diff_%d" % n, q, i)

            self.write_actions(episodic_results['a_agent'], episodic_results['a_player'], test_results['q'], n)

            self.writer.add_histogram("actions/q_diff", test_results['q_diff'], n, 'doane')
            self.writer.add_histogram("actions/a_best", test_results['a_best'], n, 'doane')
            self.writer.add_histogram("data/probabilities", agent.train_dataset.probabilities, n, 'fd')
            self.writer.add_histogram("data/td_errors", agent.train_dataset.td_error, n, 'fd')
            self.writer.add_histogram("data/td_errors", agent.train_dataset.td_error, n, 'fd')

            r = [str(i) for i in test_results['r'].tolist()]
            self.writer.add_embedding(test_results['q'][-self.batch:], metadata=r[-self.batch:], label_img=test_results['s'][:, :-1, :, :],
                                      global_step=n, tag='states_embedding')

            # log to numpy objects
            # filename = os.path.join(self.results_dir, "%d" % n)
            # np.savez(filename, train_results=train_results, test_results=test_results)

            # save agent state
            agent.save_checkpoint(self.checkpoint)

            if abs(avg_test_q_diff) < best:
                best = abs(avg_test_q_diff)
                agent.save_checkpoint(self.checkpoint_best)

            # init interval timer
            t_interval = time.time() - t_last
            t_last = time.time()

            true_score = avg_score * (human_score - random_score) + random_score
            logger.info(" ")
            #  log to screen and logger
            logger.info("------------Checkpoint @ Experiment: %s------------" % self.exp_name)
            logger.info("Evaluation @ n=%d, step=%d | Test Loss: %g |  Train Loss: %g | Avg. Score: %g (%g)"
                        % (n, train_results['n'][-1], avg_test_loss, avg_train_loss, avg_score, true_score))
            logger.info("Total Simulation time: %g| Interval time: %g| Train: %g |  Test: %g | Play: %g"
                        % (simulation_time, t_interval, t_train, t_eval, t_play))

            logger.info(" ")
            logger.info("Layers statistics:")
            for name, module in list(agent.model.named_modules()):
                if "." not in name:
                    continue
                logger.info(" ")
                logger.info("%s:" % name.split(".")[-1])
                logger.info("weights norm: %s" % str(module.weight.data.norm()))
                logger.info("bias norm: %s" % str(module.bias.data.norm()))
                logger.info("weights-grad norm: %s" % str(module.weight.grad.data.norm()))
                logger.info("bias-grad norm: %s" % str(module.bias.grad.data.norm()))

        return agent.model.state_dict()

    def write_actions(self, a_agent, a_player, q, n):

        # histogram for each action
        for activation in range(self.action_space):

            a = activation
            name = self.action_meanings[a]
            self.writer.add_histogram("actions/q_histogram_%s(%d)" % (name, a), q[:, activation], n, 'fd')

        # histogram for the normalized max action ** 2 / sum actions ** 2

        # q_square = q ** 2
        q_square = q
        max_action, _ = q_square.max(1)
        sum_action = q_square.sum(1)

        self.writer.add_histogram("actions/mormalized_max_action", max_action / sum_action, n, 'fd')

        img = a_agent
        img = convert_screen_to_rgb(img, resize=True)
        self.writer.add_image('actions/agent_action_trajectory', img, n)

        # create one-hot image of the player actions
        img = self.one_hots.index_select(0, a_player)
        img = convert_screen_to_rgb(img, resize=True)
        self.writer.add_image('actions/player_action_trajectory', img, n)

    def train(self):
        pass

    def test(self, params):

        agent = TestAgent()
        # load model
        if params is not None:
            agent.resume(params)
        elif self.load_last:
            agent.resume(self.checkpoint)
        elif self.load_best:
            agent.resume(self.checkpoint_best)

        player = agent.play(args.test_episodes)

        step = next(player)
        s, a, r, t = step
        imgplot = plt.imshow(s)
        plt.show(block=False)
        #
        for step in player:

            s, a, r, t = step
            imgplot.set_data(s)
            plt.show(block=False)
            time.sleep(1 / 20.)
            # plt.pause(1e-17)
            # time.sleep(0.1)




