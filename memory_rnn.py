import torch.utils.data
import numpy as np
import torch
import os
import parse
from tqdm import tqdm
import cv2

from logger import logger
from config import consts, args
from preprocess import preprocess_screen


def preprocess_demonstrations():
    if os.path.isdir(os.path.join(args.indir, 'data')):
        if os.path.isfile(
                os.path.join(args.indir, 'data', "%s_data.npz" % args.game)) and not args.reload_data:
            data = np.load(os.path.join(args.indir, 'data', "%s_data.npz" % args.game))
            meta = np.load(os.path.join(args.indir, 'data', "%s_meta.npz" % args.game))

            meta = {k: meta[k] for k in meta}
            data = {k: data[k] for k in data}

            meta['screens_dir'] = str(meta['screens_dir'])
            meta['trajectories_dir'] = str(meta['trajectories_dir'])

            return meta, data
    else:
        os.mkdir(os.path.join(args.indir, 'data'))

    logger.info("Parse trajectories into dict")

    pattern = "{:d},{:d},{:d},{:d},{:d}"
    data = {}
    flat = []
    meta = {'n_states': 0, 'n_epochs': 0,
            'trajectories_dir': os.path.join(args.indir, 'trajectories', args.game),
            'screens_dir': os.path.join(args.indir, 'screens', args.game),
            'frame': 0, 'reward': 1, 'score': 2, 'terminal': 3, 'action': 4, 'terminal_states': [], 'episodes': []}

    for file in tqdm(os.listdir(meta['trajectories_dir'])):
        with open(os.path.join(meta['trajectories_dir'], file)) as fo:
            trajectory = fo.read(-1)

        # two first lines are meta data and parameters names
        trajectory = trajectory.split("\n")[2:]
        trajectory = np.array([parse.parse(pattern, i).fixed for i in trajectory if i is not ""],
                              dtype=np.int)

        # we always keep one previous frame for image preprocessing
        # trajectory_skip = trajectory[args.skip-1::args.skip, :]
        trajectory_skip = trajectory[::args.skip, :]
        epoch = file.split(".")[0]
        meta[epoch] = trajectory_skip.shape[0]
        meta['n_states'] += meta[epoch]
        meta['n_epochs'] += 1
        meta['terminal_states'].append(meta['n_states'] - 1)

        meta['episodes'].append(int(epoch))

        # concatenate actions of skip number consecutive steps
        actions = trajectory[:, meta['action']]

        # # A a hack to get the skip - mod and the correct inverse sign quotient
        m = np.mod(-actions.size, args.skip)
        q, _ = np.divmod(actions.size, args.skip)

        actions = np.concatenate((actions, np.zeros(m, dtype=np.int)))
        actions = actions.reshape((q + int(m > 0), args.skip))
        # actions = actions[:q * args.skip].reshape((q, args.skip))

        # actions = np.transpose(actions)

        # add consecutive actions to trajectory skip array
        trajectory_skip = np.concatenate((trajectory_skip[:, :-1], actions), axis=1)

        # fix the last state termination - we take terminal value from the true last state before applying the skip
        trajectory_skip[-1, meta['terminal']] = trajectory[-1, meta['terminal']]

        num = int(epoch)
        # save epoch and location of flat index
        # [epoch_number, frame_number, state_index, terminal]
        # notice that: state_index = frame_number / skip
        flat.append(np.array([num * np.ones(meta[epoch], dtype=np.uint32), trajectory_skip[:, meta['frame']],
                              np.arange(meta[epoch]), trajectory_skip[:, meta['terminal']]]))

        data[epoch] = trajectory_skip
        # if meta['n_epochs'] >= 5:
        #     break

    flat = np.hstack(flat)
    flat = np.swapaxes(flat, 0, 1)
    meta['flat'] = flat
    meta['terminal_states'] = np.array(meta['terminal_states'])

    # get average score (temporal implementation)
    scores = []
    for index in meta['terminal_states']:
        epoch, frame, local, terminal = meta['flat'][index]
        trajectory = data[str(epoch)]
        scores.append(trajectory[local, meta['score']])

    meta['avg_score'] = np.mean(scores)

    np.savez(os.path.join(args.indir, 'data', "%s_data" % args.game), **data)
    np.savez(os.path.join(args.indir, 'data', "%s_meta" % args.game), **meta)

    return meta, data


def divide_dataset(meta):
    # set train and test datasets
    n = meta['n_states']
    indexes = range(n)

    # filtered_states = [i + meta['terminal_states'] for i in range(args.history_length + 1)]
    filtered_states = [i + range(-1, args.history_length) for i in meta['terminal_states']]
    # filtered_states = np.concatenate(filtered_states)

    filtered_states = np.concatenate((np.arange(args.history_length - 1), *filtered_states))
    # filter terminal states and init states with no history
    indexes = [i for i in indexes if i not in filtered_states]

    p1 = args.test_percentage
    p2 = args.test_percentage
    set_type = np.random.choice([0, 1, 2], size=n, p=[1 - p1 - p2, p1, p2])

    meta['train'] = np.array([i for i in indexes if set_type[i] == 0])
    meta['val'] = np.array([i for i in indexes if set_type[i] == 1])
    meta['test'] = np.array([i for i in indexes if set_type[i] == 2])
    meta['full'] = np.array(indexes)

    meta['reverse_full'] = np.zeros(n, dtype=int)
    meta['reverse_full'][meta['full']] = np.arange(len(meta['full']))

    meta['reverse_train'] = np.zeros(n, dtype=int)
    meta['reverse_train'][meta['train']] = np.arange(len(meta['train']))

    meta['reverse_val'] = np.zeros(n, dtype=int)
    meta['reverse_val'][meta['val']] = np.arange(len(meta['val']))

    meta['reverse_test'] = np.zeros(n, dtype=int)
    meta['reverse_test'][meta['test']] = np.arange(len(meta['test']))

    return meta


def init_demonstration_memory(cls):
    cls.update_n_step(0)
    return cls


@init_demonstration_memory
class DemonstrationRNNMemory(torch.utils.data.Dataset):

    update_interval = args.update_n_steps_interval
    n_steps = args.n_steps
    # n_steps = args.n_steps if args.bootstrap else np.Inf
    var_lambda = args.var_lambda
    # bootstrap = args.bootstrap

    def __init__(self, name, meta, data):
        super(torch.utils.data.Dataset, self).__init__()

        self.meta = meta
        self.data = data

        self.clip = np.array([args.clip if args.clip > 0 else np.inf])
        self.actions_transform = np.array(consts.action2activation[args.game])

        self.indices = self.meta[name]
        self.n = int(self.indices.size)
        self.probabilities = np.ones(self.n) / self.n
        self.td_error = np.zeros(self.n)
        self.td_discount = args.td_discount
        self.reverse_indices = self.meta['reverse_%s' % name]

        # parameters
        self.omega = args.omega
        self.skip = args.skip
        self.history_length = args.history_length
        self.discount = args.discount
        self.myopic = args.myopic
        self.action_space = consts.action_space
        self.excitation_map = consts.excitation_map
        self.mask = np.array(consts.excitation_mask[args.game], dtype=np.float32)

        self.one = np.ones(1, dtype=np.int)[0]

        # set action management according to configuration
        if args.input_actions:
            self.get_action = self.get_action_excitation
        else:
            self.get_action = self.get_action_index

    def __len__(self):
        return len(self.data)

    @classmethod
    def update_n_step(cls, i):
        if cls.var_lambda:
            # CHANGE
            cls.n_steps = int(8196 * 2 ** (-float(i) / (cls.update_interval * 2)))
            cls.n_steps = max(cls.n_steps, 1)

        # if not cls.bootstrap:
        #     cls.n_steps = np.Inf
        # if not var_lambda, the n_steps parameter does not changes

    def update_td_error(self, indices, loss):
        indices = self.reverse_indices[indices]
        self.td_error[indices] = self.td_error[indices] * (1 - self.td_discount) + self.td_discount * (loss ** self.omega)

    def update_probabilities(self):

        m = self.td_error.max()
        m = 1 if not m else m
        self.probabilities = self.td_error
        self.probabilities[self.td_error == 0] = m
        self.probabilities = self.probabilities / self.probabilities.sum()

    def get_action_excitation(self, a):
        m = np.zeros((self.action_space, self.skip))
        m[a, range(self.skip)] = 1
        m = m.sum(1)
        a = (1 + np.argmax(m[1:])) * (a.sum() != 0)
        a_vec = np.array(self.excitation_map[a], dtype=np.float32) * self.mask
        return a_vec, a

    def get_action_index(self, a):
        m = np.zeros((self.action_space, self.skip))
        m[a, range(self.skip)] = 1
        m = m.sum(1)
        a = (1 + np.argmax(m[1:])) * (a.sum() != 0)
        # transform a to a valid activation
        a = self.actions_transform[a]
        return a, a

    def __getitem__(self, index):

        epoch = index

        trajectory = self.data[str(epoch)]
        n = len(trajectory)

        frame0 = [os.path.join(self.meta['screens_dir'], str(epoch), "%d.png" % (i * self.skip)) for i in range(n-1)]
        frame1 = [os.path.join(self.meta['screens_dir'], str(epoch), "%d.png" % (1 + i * self.skip)) for i in range(n-1)]
        imgs = [preprocess_screen((f0, f1)) for f0, f1 in zip(frame0, frame1)]

        o = [np.stack(imgs[i: i - self.history_length: -1], axis=0).expand_dims(axis=0) for i in range(self.history_length-1, n-1)]
        o = np.concatenate(o)

        a = [self.get_action(actions)[0] for actions in trajectory[self.history_length-1: n-1, self.meta['action']:self.meta['action'] + self.skip]]

        # get trajectory reward
        score_pre = trajectory[self.history_length-1: n-1, self.meta['score']]
        score_post = trajectory[self.history_length, n, self.meta['score']]
        rewards = score_post - score_pre

        rewards = np.clip(rewards, -self.clip, self.clip)
        r = (rewards / self.meta['avg_score']).astype(np.float32)

        # get final score discounted from each observation

        f = [np.multiply(rewards[i:], self.discount ** np.arange(len(rewards[i:]))).sum() for i in range(n-self.history_length)]
        f = np.concatenate(f)
        f = f / self.meta['avg_score']

        return {'o': o, 'a': a, 'r': r.astype(np.float32), 'f': f.astype(np.float32)}


class SequentialDemonstrationRNNSampler(torch.utils.data.sampler.Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):

        for epoch in self.data_source.data:
            yield int(epoch)

    def __len__(self):
        return len(self.data_source)
