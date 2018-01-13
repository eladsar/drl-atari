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

from logger import logger
from preprocess import convert_screen_to_rgb



class Experiment(object):

    def __init__(self):

        # parameters
        self.input_actions = args.input_actions
        if self.input_actions:
            self.action_space = consts.action_space
        else:
            self.action_space = consts.n_actions[args.game]

        self.action_meanings = consts.action_meanings
        self.activation2action = consts.activation2action[args.game]
        self.one_hots = torch.sparse.torch.eye(self.action_space)
        self.batch = args.batch
        self.l1_loss = args.l1_loss

        dirs = os.listdir(args.outdir)

        if args.comet:
            self.comet = comet.Experiment(api_key=consts.api_key)
            self.comet.log_multiple_params(vars(args))

        self.load_model = args.load_last_model or args.load_best_model
        self.load_last = args.load_last_model
        self.load_best = args.load_best_model

        self.exp_name = ""
        if self.load_model:
            if args.resume >= 0:
                for d in dirs:
                    if "%s_exp_%04d_" % (args.identifier, args.resume) in d:
                        self.exp_name = d
                        break
            else:
                max = 0
                for d in dirs:
                    n = int(d.split("_")[2])
                    if n > max:
                        self.exp_name = d
                        break

        if not self.exp_name:
            # count similar experiments
            n = sum([1 for d in dirs if "%s_exp" % args.identifier in d])
            self.exp_name = "%s_exp_%04d_%s" % (args.identifier, n, consts.exptime)
            self.load_model = False

        # init experiment parameters
        self.root = os.path.join(args.outdir, self.exp_name)

        # set dirs
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')

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

        # initialize tensorboard writer
        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=args.identifier)
        # write csv file of hyper-parameters
        filename = os.path.join(self.root, "hyperparameters.csv")
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.exp_name)
            for k, v in vars(args).items():
                spamwriter.writerow([k, str(v)])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()

    def behavioral(self):
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
            agent.save_checkpoint(self.checkpoint, {'n': n+1})

            if abs(avg_test_loss_v) < best:
                best = abs(avg_test_loss_v)
                agent.save_checkpoint(self.checkpoint_best, {'n': n+1})

            # init interval timer
            t_interval = time.time() - t_last
            t_last = time.time()

            img = test_results['s'][0, :-1, :, :]

            self.writer.add_image('states/state', img, n)

            logger.info(" ")
            #  log to screen and logger
            logger.info("------------Checkpoint @ Behavioral Experiment: %s------------" % self.exp_name)
            logger.info("Evaluation @ n=%d, step=%d | Test Q Loss: %g | Train Q Loss: %g | Train V Loss: %g | Train beta Loss: %g"
                        % (n + n_offset, train_results['n'][-1], avg_test_loss_q, avg_train_loss_q, avg_train_loss_v, avg_train_loss_b))
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

        if args.hot:
            agent = BehavioralHotAgent(load_dataset=False)
        else:
            agent = BehavioralAgent(load_dataset=False)
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

    def play_behavioral(self, params=None):

        if args.hot:
            agent = BehavioralHotAgent(load_dataset=False)
        else:
            agent = BehavioralAgent(load_dataset=False)
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

            if not self.input_actions:
                a = self.activation2action[activation]
            else:
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




