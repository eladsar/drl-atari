import argparse


# consts
class Consts(object):
    frame_rate = 1. / 60.
    action_space = 18
    gym_game_dict = {"spaceinvaders": "SpaceInvaders-v0",
                     "mspacman": "MsPacman-v0",
                     "pinball": "VideoPinball-v0",
                     "qbert": "Qbert-v0",
                     "revenge": "MontezumaRevenge-v0"}
    nop = 0

    action_meanings = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
                       'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE',
                       'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE',
                       'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

    actions_dict = {"spaceinvaders": {'NOOP': 'NOOP',
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
                                       'DOWNLEFTFIRE': 'LEFTFIRE'}}

    actions_transform = {}
    actions_mask = {}
    action2activation = {}
    activation2action = {}
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

consts = Consts()
parser = argparse.ArgumentParser(description='Rainbow')

def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})


# Arguments
parser.add_argument('--game', type=str, default='spaceinvaders', help='ATARI game')
parser.add_argument('--identifier', type=str, default='next_model', help='The name of the model to use')
parser.add_argument('--screensdir', type=str, default='/dev/shm/sarafie/', help='Demonstration directory')
parser.add_argument('--indir', type=str, default='/data/sarafie/atari/atari_v2_release', help='Demonstration directory')

# booleans
boolean_feature("load-model", False, 'Load the saved model if possible')
boolean_feature("lfd", True, 'Train the model with demonstrations')
boolean_feature("test", False, 'Test the learned model')
boolean_feature("render", False, 'Render tested episode')
boolean_feature("reuse-preprocessed-data", True, "Evaluate only")
boolean_feature("cuda", True, "Use GPU environment for testing")
boolean_feature("train", False, "Train Reinforcement learning agent")

# parameters
parser.add_argument('--skip', type=int, default=4, help='Skip pattern')
parser.add_argument('--height', type=int, default=84, help='Image Height')
parser.add_argument('--width', type=int, default=84, help='Image width')
parser.add_argument('--batch', type=int, default=32, help='Mini-Batch Size')
parser.add_argument('--eval-batch', type=int, default=512, help='Evaluation Batch Size')
parser.add_argument('--cpu-workers', type=int, default=36, help='How many CPUs will be used for the data loading')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--test-percentage', type=float, default=0.02, help='Percentage of test dataset')
parser.add_argument('--T-max', type=int, default=int(5e7), metavar='STEPS', help='Number of training steps')
parser.add_argument('--test-episodes', type=int, default=32, metavar='STEPS', help='Number of test episodes')
parser.add_argument('--clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--evaluation-interval', type=int, default=512, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--update-interval', type=int, default=4096, metavar='STEPS', help='Number of steps after which to update target network')
parser.add_argument('--save-interval', type=int, default=32, metavar='STEPS', help='Number of steps after which to save model')
parser.add_argument('--max-episode-length', type=int, default=8192, metavar='STEPS', help='Maximum frame length of a single episode')
parser.add_argument('--n-steps', type=int, default=32, metavar='STEPS', help='Number of steps for multi-step learning')


args = parser.parse_args()