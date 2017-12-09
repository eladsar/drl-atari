import os
from tqdm import tqdm

from config import consts, args
from agent import Agent
from preprocess import get_trajectories_dict


def set_tensorboard():
    if not os.path.isdir(os.path.join(args.indir, 'tensorboard')):
        os.mkdir(os.path.join(args.indir, 'tensorboard'))


def main():

    # print args of current run
    print("Welcome to Learning from Demonstration simulation")
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    data, meta = get_trajectories_dict()
    set_tensorboard()
    agent = Agent(data, meta)

    if args.lfd:
        print("Enter LfD Session, it might take a while")
        agent.lfd()

    if args.train:
        print("Enter Training Session, it might take ages")
        agent.train()

    if args.test:
        print("Enter Testing Session, it should be fun")
        agent.test()

    print("End of simulation")


if __name__ == "__main__":
    main()

