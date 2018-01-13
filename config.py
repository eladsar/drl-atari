import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='atari')


def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})


# Arguments

# strings
parser.add_argument('--game', type=str, default='spaceinvaders', help='ATARI game')
parser.add_argument('--identifier', type=str, default='debug_behavioral', help='The name of the model to use')
parser.add_argument('--indir', type=str, default='/dev/shm/sarafie/atari/', help='Demonstration directory')
parser.add_argument('--outdir', type=str, default='/local_data/sarafie/atari/results', help='Output directory')
parser.add_argument('--logdir', type=str, default='/local_data/sarafie/atari/logs', help='Logs directory')

# booleans
boolean_feature("load-last-model", False, 'Load the last saved model if possible')
boolean_feature("load-best-model", False, 'Load the best saved model if possible')
boolean_feature("lfd", False, 'Train the model with demonstrations')
boolean_feature("behavioral", True, 'Find behavioral policies parameters')
boolean_feature("play-behavioral", False, 'Play with the behavioral policy')
boolean_feature("play", True, 'Test the learned model via playing')
boolean_feature("render", False, 'Render tested episode')
boolean_feature("reload-data", False, "Ignore loaded trajectories and load again")
boolean_feature("cuda", True, "Use GPU environment for testing")
boolean_feature("train", False, "Train Reinforcement learning agent")
boolean_feature("comet", False, "Log results to comet")
boolean_feature("tensorboard", True, "Log results to tensorboard")

# Rainbow
boolean_feature("noisynet", False, "Add noise NN for exploration")
boolean_feature("dueling", True, "Dueling neural network architecture")
boolean_feature("double-q", True, "Double Q-learning")
boolean_feature("distributional-rl", False, "Learn the Q-function distribution")
boolean_feature("prioritized-replay", False, "Sample with priority from the memory, otherwise, "
                                             "Sample from memory in sequential order without replacement")
parser.add_argument('--n-steps', type=int, default=1, metavar='STEPS', help='Number of steps for multi-step learning')

# my models parameters
boolean_feature("var-lambda", False, "Variational n_step version [overrun n-steps]")
boolean_feature("on-policy", False, "Use On-Policy evaluation for Q-learning [overrun double q-learning]")
boolean_feature("value-advantage", True, "Use Value-Advantage architecture")
boolean_feature("value-only", False, "Train only the value network in value-advantage architecture")
boolean_feature("myopic", False, "Use myopic approach for value evaluation")
boolean_feature("input-actions", True, "Use actions as input architecture")
boolean_feature("l1-loss", True, "Use L1 loss function")
boolean_feature("bootstrap", False, "Use bootstraping for value/q evaluation (if False, then n-steps=Inf)")
boolean_feature("deterministic", False, "Model a deterministic policy")
boolean_feature("imitation", False, "Try to imitate the players behavior (value function agnostic)")
boolean_feature("hot", True, "Use hot vector actions instead of excitation")

# player
boolean_feature("greedy", False, "Use greedy policy")

# parameters
parser.add_argument('--resume', type=int, default=-1, help='Resume experiment number, set -1 for last experiment')

# model parameters
parser.add_argument('--skip', type=int, default=2, help='Skip pattern')
parser.add_argument('--height', type=int, default=84, help='Image Height')
parser.add_argument('--width', type=int, default=84, help='Image width')
parser.add_argument('--batch', type=int, default=32, help='Mini-Batch Size')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--update-target-interval', type=int, default=4096*8, metavar='STEPS', help='Number of traning iterations between q-target updates')
parser.add_argument('--update-memory-interval', type=int, default=4096*4, metavar='STEPS', help='Number of steps between memory updates')
parser.add_argument('--update-n-steps-interval', type=int, default=4096*4, metavar='STEPS', help='Number of steps between memory updates')
parser.add_argument('--omega', type=float, default=0.5, metavar='ω', help='Attenuation factor for the prioritized replay distribution')
parser.add_argument('--td-discount', type=float, default=0.5, metavar='discount', help='The discount factor for calculating the TD-error')
parser.add_argument('--margin', type=float, default=0.8, help='Large margin offset for learning from demonstrations')
parser.add_argument('--final-score-reward', type=float, default=0.05, help='Large margin offset for learning from demonstrations')


# distributional RL
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')

# dataloader
parser.add_argument('--cpu-workers', type=int, default=32, help='How many CPUs will be used for the data loading')
parser.add_argument('--gpu-workers', type=int, default=8, help='How many parallel processes feeds the GPUs in play mode')
parser.add_argument('--cuda-default', type=int, default=2, help='Default GPU')

# train parameters
parser.add_argument('--val-percentage', type=float, default=0.00, help='Percentage of validation dataset')
parser.add_argument('--test-percentage', type=float, default=0.08, help='Percentage of test dataset')
parser.add_argument('--n-tot', type=int, default=int(30e6), metavar='STEPS', help='Total number of training steps')
parser.add_argument('--checkpoint-interval', type=int, default=4096, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluate-frames', type=int, default=256, metavar='STEPS', help='Number of frames for test evaluation')

# test parameters
parser.add_argument('--test-episodes', type=int, default=128, metavar='STEPS', help='Number of test episodes')
parser.add_argument('--max-episode-length', type=int, default=8192, metavar='STEPS', help='Maximum frame length of a single episode')

# optimizer arguments
parser.add_argument('--optimizer', type=str, default='rmsprop', help='Optimizer type')
# for stochastic beta
parser.add_argument('--lr-beta', type=float, default=0.008, metavar='η', help='Learning rate for behavioral policy')
# for deterministic beta
# parser.add_argument('--lr-beta', type=float, default=0.0003, metavar='η', help='Learning rate for behavioral policy')
parser.add_argument('--lr', type=float, default=0.0002, metavar='η', help='Learning rate')
parser.add_argument('--lr-q', type=float, default=0.0002, metavar='η', help='Learning rate for behavioral q-value')
parser.add_argument('--lr-p', type=float, default=0.00002, metavar='η', help='Learning rate for behavioral predict net')
parser.add_argument('--lr-r', type=float, default=1e-6, metavar='η', help='Learning rate for behavioral reward function')
parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='ε', help='Adam epsilon')
parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='beta', help='Adam coefficients used for computing running averages of gradient')
parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='beta', help='Adam coefficients used for computing running averages of gradient')
parser.add_argument('--decay', type=float, default=0.9, metavar='gamma', help='Decay the learning rate by this factor every update target interval')

args = parser.parse_args()


# consts
class Consts(object):

    exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    api_key = "jthyXB1jO4czVy63ntyWZSnlf"

    frame_rate = 1. / 60.
    action_space = 18
    gym_game_dict = {"spaceinvaders": "SpaceInvadersNoFrameskip-v4",
                     "mspacman": "MsPacmanNoFrameskip-v4",
                     "pinball": "VideoPinballNoFrameskip-v4",
                     "qbert": "QbertNoFrameskip-v4",
                     "revenge": "MontezumaRevengeNoFrameskip-v4"}

    flicker = {"spaceinvaders": True,
                     "mspacman": True,
                     "pinball": False,
                     "qbert": False,
                     "revenge": False}

    # random_frames = {"spaceinvaders": args.skip * ,
    #                  "mspacman": True,
    #                  "pinball": False,
    #                  "qbert": False,
    #                  "revenge": False}


    nop = 0

    action_meanings = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
                       'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE',
                       'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE',
                       'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

    actions_dict = {"revenge": {'NOOP': 'NOOP',
                                       'FIRE': 'FIRE',
                                       'UP': 'UP',
                                       'RIGHT': 'RIGHT',
                                       'LEFT': 'LEFT',
                                       'DOWN': 'DOWN',
                                       'UPRIGHT': 'UPRIGHT',
                                       'UPLEFT': 'UPLEFT',
                                       'DOWNRIGHT': 'DOWNRIGHT',
                                       'DOWNLEFT': 'DOWNLEFT',
                                       'UPFIRE': 'UPFIRE',
                                       'RIGHTFIRE': 'RIGHTFIRE',
                                       'LEFTFIRE': 'LEFTFIRE',
                                       'DOWNFIRE': 'DOWNFIRE',
                                       'UPRIGHTFIRE': 'UPRIGHTFIRE',
                                       'UPLEFTFIRE': 'UPLEFTFIRE',
                                       'DOWNRIGHTFIRE': 'DOWNRIGHTFIRE',
                                       'DOWNLEFTFIRE': 'DOWNLEFTFIRE'},
                    "spaceinvaders": {'NOOP': 'NOOP',
                                      'FIRE': 'FIRE',
                                      'UP': 'NOOP',
                                      'RIGHT': 'RIGHT',
                                      'LEFT': 'LEFT',
                                      'DOWN': 'NOOP',
                                      'UPRIGHT': 'RIGHT',
                                      'UPLEFT': 'LEFT',
                                      'DOWNRIGHT': 'RIGHT',
                                      'DOWNLEFT': 'LEFT',
                                      'UPFIRE': 'FIRE',
                                      'RIGHTFIRE': 'RIGHTFIRE',
                                      'LEFTFIRE': 'LEFTFIRE',
                                      'DOWNFIRE': 'FIRE',
                                      'UPRIGHTFIRE': 'RIGHTFIRE',
                                      'UPLEFTFIRE': 'LEFTFIRE',
                                      'DOWNRIGHTFIRE': 'RIGHTFIRE',
                                      'DOWNLEFTFIRE': 'LEFTFIRE'},
                    "mspacman": {'NOOP': 'NOOP',
                                      'FIRE': 'NOOP',
                                      'UP': 'UP',
                                      'RIGHT': 'RIGHT',
                                      'LEFT': 'LEFT',
                                      'DOWN': 'DOWN',
                                      'UPRIGHT': 'UPRIGHT',
                                      'UPLEFT': 'UPLEFT',
                                      'DOWNRIGHT': 'DOWNRIGHT',
                                      'DOWNLEFT': 'DOWNLEFT',
                                      'UPFIRE': 'UP',
                                      'RIGHTFIRE': 'RIGHT',
                                      'LEFTFIRE': 'LEFT',
                                      'DOWNFIRE': 'DOWN',
                                      'UPRIGHTFIRE': 'UPRIGHT',
                                      'UPLEFTFIRE': 'UPLEFT',
                                      'DOWNRIGHTFIRE': 'DOWNRIGHT',
                                      'DOWNLEFTFIRE': 'DOWNLEFT'}
                    }

    actions_transform = {}
    actions_mask = {}
    action2activation = {}
    activation2action = {}
    n_actions = {}

    # IT MIGHT BE A WRONG IMPLEMENTATION !!!!!!!
    for game in actions_dict:
        actions_transform[game] = [0] * action_space
        actions_mask[game] = [0] * action_space
        for a in actions_dict[game]:
            i = action_meanings.index(a)
            j = action_meanings.index(actions_dict[game][a])
            actions_transform[game][i] = j
            actions_mask[game][j] = 1

        # strangely it output NameError in python 3.5 but not in python 3.6
        # activation2action[game] = [actions_transform[game][i] for i in range(action_space) if actions_mask[game][i]]
        # action2activation[game] = [activation2action[game].index(actions_transform[game][i]) for i in range(action_space)]

        activation2action[game] = []
        action2activation[game] = [0] * action_space
        for i in range(action_space):
            if actions_mask[game][i]:
                activation2action[game].append(actions_transform[game][i])
            action2activation[game][i] = activation2action[game].index(actions_transform[game][i])

        n_actions[game] = len(activation2action[game])

    actions_mask = {"spaceinvaders": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    }

    random_score = {"spaceinvaders": 156,
                     "mspacman": 211,
                     "pinball": 30368,
                     "qbert": 162,
                     "revenge": 0}

    # construct excitation vectors
    excitation_map = np.zeros((action_space, 3), dtype=np.int)

    for i in range(action_space):
        name = action_meanings[i]
        if 'FIRE' in name:
            excitation_map[i, 0] = 1
        if 'UP' in name:
            excitation_map[i, 1] = 1
        elif 'DOWN' in name:
            excitation_map[i, 1] = -1
        if 'RIGHT' in name:
            excitation_map[i, 2] = 1
        elif 'LEFT' in name:
            excitation_map[i, 2] = -1

    reverse_excitation_map = {
        (0, 0, 0): 'NOOP',
        (0, 0, 1): 'RIGHT',
        (0, 0, -1): 'LEFT',
        (0, 1, 0): 'UP',
        (0, 1, 1): 'UPRIGHT',
        (0, 1, -1): 'UPLEFT',
        (0, -1, 0): 'DOWN',
        (0, -1, 1): 'DOWNRIGHT',
        (0, -1, -1): 'DOWNLEFT',
        (1, 0, 0): 'FIRE',
        (1, 0, 1): 'RIGHTFIRE',
        (1, 0, -1): 'LEFTFIRE',
        (1, 1, 0): 'UPFIRE',
        (1, 1, 1): 'UPRIGHTFIRE',
        (1, 1, -1): 'UPLEFTFIRE',
        (1, -1, 0): 'DOWNFIRE',
        (1, -1, 1): 'DOWNRIGHTFIRE',
        (1, -1, -1): 'DOWNLEFTFIRE',
    }
    reverse_excitation_index = {
        (0, 0, 0): 0,
        (0, 0, 1): 3,
        (0, 0, -1): 4,
        (0, 1, 0): 2,
        (0, 1, 1): 6,
        (0, 1, -1): 7,
        (0, -1, 0): 5,
        (0, -1, 1): 8,
        (0, -1, -1): 9,
        (1, 0, 0): 1,
        (1, 0, 1): 11,
        (1, 0, -1): 12,
        (1, 1, 0): 10,
        (1, 1, 1): 14,
        (1, 1, -1): 15,
        (1, -1, 0): 13,
        (1, -1, 1): 16,
        (1, -1, -1): 17,
    }


    excitation_mask = {"spaceinvaders": [1, 0, 1],
                       "mspacman": [0, 1, 1],
                       "pinball": [1, 1, 1],
                       "qbert": [0, 1, 1],
                       "revenge": [1, 1, 1]}

    #                 n, f, u, n, d, l, n, r
    hotvec_matrix = [[1, 0, 0, 1, 0, 0, 1, 0],  # NOOP
                     [0, 1, 0, 1, 0, 0, 1, 0],  # FIRE
                     [1, 0, 1, 0, 0, 0, 1, 0],  # UP
                     [1, 0, 0, 1, 0, 0, 0, 1],  # RIGHT
                     [1, 0, 0, 1, 0, 1, 0, 0],  # LEFT
                     [1, 0, 0, 0, 1, 0, 1, 0],  # DOWN
                     [1, 0, 1, 0, 0, 0, 0, 1],  # UPRIGHT
                     [1, 0, 1, 0, 0, 1, 0, 0],  # UPLEFT
                     [1, 0, 0, 0, 1, 0, 0, 1],  # DOWNRIGHT
                     [1, 0, 0, 0, 1, 1, 0, 0],  # DOWNLEFT
                     [0, 1, 1, 0, 0, 0, 1, 0],  # UPFIRE
                     [0, 1, 0, 1, 0, 0, 0, 1],  # RIGHTFIRE
                     [0, 1, 0, 1, 0, 1, 0, 0],  # LEFTFIRE
                     [0, 1, 0, 0, 1, 0, 1, 0],  # DOWNFIRE
                     [0, 1, 1, 0, 0, 0, 0, 1],  # UPRIGHTFIRE
                     [0, 1, 1, 0, 0, 1, 0, 0],  # UPLEFTFIRE
                     [0, 1, 0, 0, 1, 0, 0, 1],  # DOWNRIGHTFIRE
                     [0, 1, 0, 0, 1, 1, 0, 0]   # DOWNLEFTFIRE
                     ]
    hotvec_inv   = {(1, 0, 0, 1, 0, 0, 1, 0): 0,  # NOOP
                    (0, 1, 0, 1, 0, 0, 1, 0): 1,  # FIRE
                    (1, 0, 1, 0, 0, 0, 1, 0): 2,  # UP
                    (1, 0, 0, 1, 0, 0, 0, 1): 3,  # RIGHT
                    (1, 0, 0, 1, 0, 1, 0, 0): 4,  # LEFT
                    (1, 0, 0, 0, 1, 0, 1, 0): 5,  # DOWN
                    (1, 0, 1, 0, 0, 0, 0, 1): 6,  # UPRIGHT
                    (1, 0, 1, 0, 0, 1, 0, 0): 7,  # UPLEFT
                    (1, 0, 0, 0, 1, 0, 0, 1): 8,  # DOWNRIGHT
                    (1, 0, 0, 0, 1, 1, 0, 0): 9,  # DOWNLEFT
                    (0, 1, 1, 0, 0, 0, 1, 0): 10,  # UPFIRE
                    (0, 1, 0, 1, 0, 0, 0, 1): 11,  # RIGHTFIRE
                    (0, 1, 0, 1, 0, 1, 0, 0): 12,  # LEFTFIRE
                    (0, 1, 0, 0, 1, 0, 1, 0): 13,  # DOWNFIRE
                    (0, 1, 1, 0, 0, 0, 0, 1): 14,  # UPRIGHTFIRE
                    (0, 1, 1, 0, 0, 1, 0, 0): 15,  # UPLEFTFIRE
                    (0, 1, 0, 0, 1, 0, 0, 1): 16,  # DOWNRIGHTFIRE
                    (0, 1, 0, 0, 1, 1, 0, 0): 17   # DOWNLEFTFIRE
                     }


    # excitation_length = {}
    #
    # for game in excitation_mask:
    #     excitation_length[game] = sum(excitation_mask[game])
    #
    # excitation_vectors = {}
    # excitation2action = {}
    # action2excitation = {}
    # for game in excitation_mask:
    #     x = np.multiply(excitation_map, np.tile(excitation_mask[game], (action_space, 1)))
    #     excitation_vectors[game] = [list(j) for j in set(tuple(i) for i in x.tolist())]
    #
    #     n = len(excitation_vectors[game])
    #     excitation2action[game] = np.zeros(n, dtype=np.int)
    #
    #     action2excitation[game] = [None] * action_space
    #     for i in range(n):
    #         vec = excitation_vectors[game][i]
    #         ind = excitation_map.tolist().index(vec)
    #         excitation2action[game][i] = ind
    #         action2excitation[game][ind] = vec
    #
    # for game in excitation_vectors:
    #     for j in range(3):
    #         if not excitation_mask[game][j]:
    #             for i in range(len(excitation_vectors[game])):
    #                 excitation_vectors[game][i].pop(j)


consts = Consts()

