import torch.multiprocessing as mp

def main():

    # print args of current run
    logger.info("Welcome to Learning from Demonstration simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    data, meta = get_trajectories_dict()

    with Experiment() as exp:

        if args.lfd:
            logger.info("Enter LfD Session, it might take a while")
            model = exp.lfd(meta, data)

        if args.train:
            logger.info("Enter Training Session, it might take ages")
            model = exp.train(model)

        if args.test:
            logger.info("Enter Testing Session, it should be fun")
            exp.test(model)

    logger.info("End of simulation")


if __name__ == '__main__':
    mp.set_start_method('forkserver')

    from config import consts, args
    from logger import logger
    from preprocess import get_trajectories_dict
    from experiment import Experiment
    main()

