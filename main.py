from config import consts, args
from logger import logger
from experiment import Experiment
import torch

def main():

    # print args of current run
    logger.info("Welcome to Learning from Demonstration simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    model = None
    with torch.cuda.device(args.cuda_default):
        with Experiment() as exp:

            if args.behavioral:
                logger.info("Enter Behavioral Learning Session, it might take a while")
                model = exp.behavioral()

            if args.play_behavioral:
                logger.info("Enter Behavioral playing, I hope it goes well")
                model = exp.play_behavioral()

            if args.lfd:
                logger.info("Enter LfD Session, it might take a while")
                model = exp.lfd()

            if args.train:
                logger.info("Enter Training Session, it might take ages")
                model = exp.train(model)

            if args.play:
                logger.info("Enter Testing Session, it should be fun")
                exp.test(model)

    logger.info("End of simulation")


if __name__ == '__main__':
    main()

