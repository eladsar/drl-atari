from config import consts, args
from logger import logger
from experiment import Experiment
import torch
from memory import preprocess_demonstrations

def main():

    # print args of current run
    logger.info("Welcome to Learning from Demonstration simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    if args.reload_data:
        preprocess_demonstrations(args.reload_data)

    model = None
    with torch.cuda.device(0 if args.parallel else args.cuda_default):

        with Experiment() as exp:

            if args.visualize:
                logger.info("Vusualize model parameters")
                model = exp.visualize('model_b')

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
                exp.play()

    logger.info("End of simulation")


if __name__ == '__main__':
    main()

