import csv
import time
import os
import sys
import numpy as np
import torch

import pandas as pd
from tensorboardX import SummaryWriter
import comet_ml as comet
import torch
from torch.autograd import Variable
import cv2
from tqdm import tqdm
import time
import torch.multiprocessing as mp

from demonstration import DemonstrationsDataset
from config import consts, args
from agent import LfdAgent
from logger import logger
from preprocess import convert_screen_to_rgb

class Experiment(object):

    def __init__(self):

        dirs = os.listdir(args.outdir)

        if args.comet:
            self.comet = comet.Experiment(api_key=consts.api_key)
            self.comet.log_multiple_params(vars(args))

        self.load_model = args.load_model

        self.exp_name = ""
        if self.load_model:
            if args.resume:
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
        self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
        self.writer.close()

    def lfd(self, meta, data):

        dataset = DemonstrationsDataset(data, meta)
        agent = LfdAgent(dataset, dataset.meta['train'], dataset.meta['test'])
        best = np.inf
        eval = agent.evaluate(args.evaluate_interval, args.n_tot)
        play = agent.play(args.test_episodes, args.n_tot)

        # save graph to tensorboard
        play_results = next(play)
        test_results = next(eval)
        s = Variable(test_results['s'], requires_grad=False)
        self.writer.add_graph(agent.model, agent.model(s))

        if self.load_model:
            agent.resume(self.checkpoint)

        for n, train_results in enumerate(agent.train(args.checkpoint_interval, args.n_tot)):

            test_results = next(eval)

            # if not n % 10:
            play_results = next(play)

            avg_train_loss = np.mean(train_results['loss'])
            avg_test_loss = np.mean(test_results['loss'])
            avg_score = np.mean(play_results['scores'])
            std_score = np.std(play_results['scores'])

            logger.info(" ")
            #  log to screen and logger
            logger.info("Evaluation @ n=%d, epoch=%d, step=%d | Avg. Test Loss: %g |  Avg. Train Loss: %g "
                        % (n, train_results['epoch'][-1], train_results['n'][-1], avg_test_loss, avg_train_loss))

            logger.info(" ")
            logger.info("Layers statistics:")
            logger.info(" ")
            logger.info("conv1:")
            logger.info("weights norm: %s" % str(agent.model.module.conv1.weight.data.norm()))
            logger.info("bias norm: %s" % str(agent.model.module.conv1.bias.data.norm()))
            logger.info("weights-grad norm: %s" % str(agent.model.module.conv1.weight.grad.data.norm()))
            logger.info("bias-grad norm: %s" % str(agent.model.module.conv1.bias.grad.data.norm()))
            logger.info(" ")
            logger.info("conv2:")
            logger.info("weights norm: %s" % str(agent.model.module.conv2.weight.data.norm()))
            logger.info("bias norm: %s" % str(agent.model.module.conv2.bias.data.norm()))
            logger.info("weights-grad norm: %s" % str(agent.model.module.conv2.weight.grad.data.norm()))
            logger.info("bias-grad norm: %s" % str(agent.model.module.conv2.bias.grad.data.norm()))
            logger.info(" ")
            logger.info("conv3:")
            logger.info("weights norm: %s" % str(agent.model.module.conv3.weight.data.norm()))
            logger.info("bias norm: %s" % str(agent.model.module.conv3.bias.data.norm()))
            logger.info("weights-grad norm: %s" % str(agent.model.module.conv3.weight.grad.data.norm()))
            logger.info("bias-grad norm: %s" % str(agent.model.module.conv3.bias.grad.data.norm()))
            logger.info(" ")
            logger.info("fc4:")
            logger.info("fc4 weights norm: %s" % str(agent.model.module.fc4.weight.data.norm()))
            logger.info("fc4 bias norm: %s" % str(agent.model.module.fc4.bias.data.norm()))
            logger.info("fc4 weights-grad norm: %s" % str(agent.model.module.fc4.weight.grad.data.norm()))
            logger.info("fc4 bias-grad norm: %s" % str(agent.model.module.fc4.bias.grad.data.norm()))
            logger.info(" ")
            logger.info("fc5:")
            logger.info("weights norm: %s" % str(agent.model.module.fc5.weight.data.norm()))
            logger.info("bias norm: %s" % str(agent.model.module.fc5.bias.data.norm()))
            logger.info("weights-grad norm: %s" % str(agent.model.module.fc5.weight.grad.data.norm()))
            logger.info("bias-grad norm: %s" % str(agent.model.module.fc5.bias.grad.data.norm()))
            logger.info(" ")

            # Log to Comet.ml
            if args.comet:
                self.comet.log_step(n)
                self.comet.log_loss(avg_test_loss)
                self.comet.log_accuracy(avg_score)

                self.comet.log_metric("test_loss", avg_test_loss)
                self.comet.log_metric("train_loss", avg_train_loss)
                self.comet.log_metric("epoch", train_results['epoch'][-1])
                self.comet.log_metric("avg_score", avg_score)
                self.comet.log_metric("std_score", std_score)

            # log to tensorboard

            self.writer.add_scalar('data/avg_train_loss', float(avg_train_loss), n)
            self.writer.add_scalar('data/avg_test_loss', float(avg_test_loss), n)
            self.writer.add_scalar('data/avg_avg_score', float(avg_score), n)
            self.writer.add_scalar('data/avg_std_score', float(std_score), n)

            img = test_results['s'][0, 0, :, :]
            img = convert_screen_to_rgb(img)

            self.writer.add_image('state', img, n)

            img = test_results['s_tag'][0, 0, :, :]
            img = convert_screen_to_rgb(img)

            self.writer.add_image('next_state', img, n)

            for name, param in agent.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), n, 'fd')

            r = [str(i) for i in test_results['r'].tolist()]
            self.writer.add_embedding(test_results['q'], metadata=r, label_img=test_results['s'][:, :-1, :, :],
                                      global_step=n, tag='states_embedding')

            # log to numpy objects
            filename = os.path.join(self.results_dir, "%d" % n)
            np.savez(filename, train_results=train_results, test_results=test_results)

            # save agent state
            agent.save_checkpoint(self.checkpoint)

            if avg_test_loss < best:
                best = avg_test_loss
                agent.save_checkpoint(self.checkpoint_best)

        return agent.model.state_dict()

    def train(self):
        pass

    def test(self):
        pass





