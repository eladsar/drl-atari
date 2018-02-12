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


# dataset v1 pattern
# pattern = "{:d},{:d},{:d},{:yesno},{:d}"
# pattern_dict = dict(yesno=lambda x: x is 'True')

def preprocess_demonstrations(reload=False):
    if os.path.isdir(os.path.join(args.indir, 'data')):
        if os.path.isfile(
                os.path.join(args.indir, 'data', "%s_data.npz" % args.game)) and not reload:
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
    meta = {'n_states': 0, 'n_epochs': 0, 'histogram': np.zeros(consts.action_space),
            'trajectories_dir': os.path.join(args.indir, 'trajectories', args.game),
            'screens_dir': os.path.join(args.indir, 'screens', args.game),
            'eol_dir': os.path.join(args.indir, 'eol', args.game),
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

        # add to histogram
        meta['histogram'] += np.bincount(actions, minlength=consts.action_space)

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
                              np.arange(meta[epoch]), trajectory_skip[:, meta['terminal']],
                              trajectory_skip[:, meta['action']]]))

        # add terminal states

        with open(os.path.join(meta['eol_dir'], file)) as terminals:
            terminals = terminals.read(-1)
        if terminals:
            terminals = terminals.split("\n")[:-1]
            for i in terminals:
                state, _ = divmod(int(i), args.skip)
                trajectory_skip[state, meta['terminal']] = 1

        data[epoch] = trajectory_skip
        # if meta['n_epochs'] >= 5:
        #     break

    flat = np.hstack(flat)
    flat = np.swapaxes(flat, 0, 1)
    meta['flat'] = flat
    meta['terminal_states'] = np.array(meta['terminal_states'])

    meta['histogram'] = meta['histogram'] / meta['histogram'].sum()

    # get average score (temporal implementation)
    scores = []
    for index in meta['terminal_states']:
        epoch, frame, local, terminal, _ = meta['flat'][index]
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

    p1 = args.val_percentage
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

    if p1 > 0:
        meta['reverse_val'] = np.zeros(n, dtype=int)
        meta['reverse_val'][meta['val']] = np.arange(len(meta['val']))

    if p2 > 0:
        meta['reverse_test'] = np.zeros(n, dtype=int)
        meta['reverse_test'][meta['test']] = np.arange(len(meta['test']))

    return meta


def divide_dataset_by_episodes(meta):
    # set train and test datasets
    n = meta['n_states']
    indexes = range(n)

    epochs = meta['n_epochs']
    test_epochs = int(round(args.test_percentage * epochs))
    last_test = meta['terminal_states'][test_epochs - 1]

    test_indices = indexes[:last_test+1]
    train_indexes = indexes[last_test+1:]

    # filtered_states = [i + meta['terminal_states'] for i in range(args.history_length + 1)]
    filtered_states = [i + range(-1, args.history_length) for i in meta['terminal_states']]
    # filtered_states = np.concatenate(filtered_states)

    filtered_states = np.concatenate((np.arange(args.history_length), *filtered_states))
    # filter terminal states and init states with no history
    test_indices = [i for i in test_indices if i not in filtered_states]
    train_indexes = [i for i in train_indexes if i not in filtered_states]

    meta['train'] = np.array(train_indexes)
    meta['test'] = np.array(test_indices)
    meta['full'] = np.array(indexes)

    meta['reverse_full'] = np.zeros(n, dtype=int)
    meta['reverse_full'][meta['full']] = np.arange(len(meta['full']))

    meta['reverse_train'] = np.zeros(n, dtype=int)
    meta['reverse_train'][meta['train']] = np.arange(len(meta['train']))

    meta['reverse_test'] = np.zeros(n, dtype=int)
    meta['reverse_test'][meta['test']] = np.arange(len(meta['test']))

    return meta


class Memory(torch.utils.data.Dataset):

    def __init__(self):
        super(Memory, self).__init__()

    def __len__(self):
        return args.n_tot

    def __getitem__(self, index):
        raise NotImplementedError


def init_demonstration_memory(cls):
    cls.update_n_step(0)
    return cls


@init_demonstration_memory
class DemonstrationMemory(Memory):

    update_interval = args.update_n_steps_interval
    n_steps = args.n_steps
    decrease = args.decrease
    # n_steps = args.n_steps if args.bootstrap else np.Inf
    var_lambda = args.var_lambda
    # bootstrap = args.bootstrap

    def __init__(self, name, meta, data):
        super(DemonstrationMemory, self).__init__()

        self.global_action_offset = args.action_offset
        self.global_reward_offset = args.reward_offset

        self.meta = meta
        self.data = data

        self.clip = np.array([args.clip if args.clip > 0 else np.inf])
        # self.actions_transform = np.array(consts.action2activation[args.game])

        self.indices = self.meta[name]
        self.n = int(self.indices.size)

        if args.balance:
            a = self.meta['flat'][:, 4]
            a = a[self.indices]
            a = np.roll(a, -self.global_action_offset)
            self.mask = np.array(consts.actions_mask[args.game])

            self.probabilities = self.meta['histogram'][a].copy()
            self.probabilities = self.mask[a] / (self.probabilities + args.balance_epsilon)

        else:
            self.probabilities = np.ones(self.n)

        self.probabilities_a = self.probabilities.copy()
        self.probabilities_b = self.probabilities.copy()

        if args.alternating:
            self.probabilities_a[int(self.n / 2):] = 0
            self.probabilities_b[:int(self.n / 2)] = 0

        self.probabilities_a = self.probabilities_a / self.probabilities_a.sum()
        self.probabilities_b = self.probabilities_b / self.probabilities_b.sum()
        self.probabilities = self.probabilities / self.probabilities.sum()

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
        self.reverse_excitation_index = consts.reverse_excitation_index
        self.mask = np.array(consts.excitation_mask[args.game], dtype=np.float32)
        self.hot = args.hot
        self.hotvec_matrix = np.array(consts.hotvec_matrix, dtype=np.float32)
        self.hotvec_int_matrix = np.array(consts.hotvec_matrix, dtype=np.int)
        self.one = np.ones(1, dtype=np.int)[0]
        self.zero = np.zeros(1, dtype=np.int)[0]
        self.horizon = args.horizon // args.skip

        self.termination_reward = np.array([args.termination_reward], dtype=np.float32)[0]

        self.horizon_action_matrix = consts.horizon_action_matrix
        self.mask_horizon_matrix = consts.mask_horizon_matrix

        # if args.recurrent:
        #     self.__getitem__ = self.get_episode
        # else:
        #     self.__getitem__ = self.get_sample

        # set action management according to configuration

        # self.get_action = self.get_hot_action
        self.get_action = self.simple_get_action

        self.get_horizon = self.get_hot_horizon
        # if args.input_actions:
        #     if self.hot:
        #         self.get_action = self.get_hot_action
        #     else:
        #         self.get_action = self.get_action_excitation
        #
        # else:
        #     self.get_action = self.get_action_index

    @classmethod
    def update_n_step(cls, i=0):
        cls.n_steps -= cls.decrease
        cls.n_steps = max(cls.n_steps, 1)
        # if cls.var_lambda:
        #     # CHANGE
        #     cls.n_steps = int(8196 * 2 ** (-float(i) / (cls.update_interval * 2)))
        #     cls.n_steps = max(cls.n_steps, 1)

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

    def simple_get_action(self, a):
        if np.random.randn() > 0:
            return a[0], a[0]
        return a[1], a[1]

    def get_hot_action(self, a):

        a0 = self.excitation_map[a[0]]
        a1 = self.excitation_map[a[1]]
        a = self.reverse_excitation_index[(a1[0], a0[1], a0[2])]
        a_vec = self.hotvec_matrix[a]
        return a_vec, a

    # def get_action_excitation(self, a):
    #     m = np.zeros((self.action_space, self.skip))
    #     m[a, range(self.skip)] = 1
    #     m = m.sum(1)
    #     a = (1 + np.argmax(m[1:])) * (a.sum() != 0)
    #     a_vec = np.array(self.excitation_map[a], dtype=np.float32) * self.mask
    #     return a_vec, a
    #
    # def get_action_index(self, a):
    #
    #     m = np.zeros((self.action_space, self.skip))
    #     m[a, range(self.skip)] = 1
    #     m = m.sum(1)
    #     a = (1 + np.argmax(m[1:])) * (a.sum() != 0)
    #     # a = a[0]
    #
    #     # transform a to a valid activation
    #     a = self.actions_transform[a]
    #     return a, a

    def get_hot_horizon(self, local, actions_index):

        a0 = max(0, local-self.horizon)
        an = min(len(actions_index), local+self.horizon)
        pad0 = (a0 - (local-self.horizon)) * self.skip
        padn = (local+self.horizon - an) * self.skip
        actions_index = actions_index[a0:an]
        actions_index = np.pad(actions_index.flatten(), (pad0, padn), "edge")
        actions = self.hotvec_int_matrix[actions_index]
        return actions, actions_index

    def get_simple_horizon(self, local, actions_index):

        an = min(len(actions_index), local+self.horizon)
        padn = (local+self.horizon - an)

        actions_index = actions_index[local:an]
        if np.random.randn() > 0:
            actions_index = np.pad(actions_index[:, 0], (0, padn), "edge")
        else:
            actions_index = np.pad(actions_index[:, 1], (0, padn), "edge")

        return actions_index, actions_index


    def get_actions_and_project(self, horizon_index):

        a_pre = horizon_index[:self.horizon*self.skip]
        a_post = horizon_index[self.horizon*self.skip:]
        last = a_pre[-1]
        # action_index_option_matrix = self.horizon_action_matrix.copy()
        # np.place(action_index_option_matrix, self.mask_horizon_matrix, last)
        # action_option_matrix = self.hotvec_int_matrix[action_index_option_matrix]

        switch = (a_post != last).nonzero()[0]
        match = 18 * 4 if not len(switch) else a_post[switch[0]] * 4 + int(round(switch[0] / 4))
        # quarter = 4 if not len(switch) else int(round(switch[0] / 4))

        match = np.array([match], dtype=np.int)
        # quarter = np.array([quarter], dtype=np.int)
        # last = np.array([last], dtype=np.int)

        return match

    def __getitem__(self, index):

        epoch, frame, local, terminal, _ = self.meta['flat'][index]
        trajectory = self.data[str(epoch)]

        # get current state parameters
        s = self.preprocess_history(epoch, frame)

        # add offsets in javatari database
        action_offset = min(self.global_action_offset, self.meta[str(epoch)] - local - 1)
        reward_offset = min(self.global_reward_offset, self.meta[str(epoch)] - local - 1)

        locala = local + action_offset
        localr = local + reward_offset

        # k steps forward
        k = min(self.n_steps, self.meta[str(epoch)] - locala - 1)

        terminal = trajectory[localr+1: localr+1 + k, self.meta['terminal']]
        terminal = np.nonzero(terminal)[0]
        if len(terminal):
            k = terminal[0] + 1
            t = self.one
        else:
            t = self.zero

        # get the most frequent action. Action NOOP is chosen only if all the series is NOOP
        actions = trajectory[locala, self.meta['action']:self.meta['action'] + self.skip]
        a, a_index = self.get_action(actions)

        # get next state parameters
        s_tag = self.preprocess_history(epoch, frame + k * self.skip)

        actions = trajectory[locala + k, self.meta['action']:self.meta['action'] + self.skip]
        a_tag, a_tag_index = self.get_action(actions)

        # get actions horizon:

        horizon, horizon_index = self.get_horizon(locala, trajectory[:, self.meta['action']:self.meta['action'] + self.skip])
        matched_option = self.get_actions_and_project(horizon_index)

        horizon_pre = horizon[: self.skip * self.horizon]
        # horizon_index_pre = horizon_index[: self.skip * self.horizon]

        horizon_tag, horizon_index_tag = self.get_horizon(locala + k, trajectory[:, self.meta['action']:self.meta['action'] + self.skip])
        # options_tag, quarter_tag, _ = self.get_actions_and_project(horizon_index_tag)
        horizon_pre_tag = horizon_tag[: self.skip * self.horizon]


        # is next state a terminal
        # t = self.one if (self.myopic or not DemonstrationMemory.bootstrap) else trajectory[local + k, self.meta['terminal']]
        # t = self.one if self.myopic else trajectory[local+1 + k, self.meta['terminal']]

        # get trajectory rewards

        score_pre = trajectory[localr:(localr + k), self.meta['score']]
        score_post = trajectory[(localr + 1):(localr + k + 1), self.meta['score']]
        rewards = score_post - score_pre

        if self.clip:
            rewards = np.clip(rewards, -self.clip, self.clip)
        else:
            rewards = rewards / self.meta['avg_score']

        if t:
            rewards[-1] = self.termination_reward

        discounts = self.discount ** np.arange(k)
        r = np.multiply(rewards, discounts).sum()

        # find monte carlo score

        terminal = trajectory[localr+1:, self.meta['terminal']]
        nxt_term = np.nonzero(terminal)[0]
        if len(nxt_term):
            terminal = nxt_term[0] + 1
        else:
            terminal = len(terminal)

        score_pre = trajectory[localr:localr+terminal, self.meta['score']]
        score_post = trajectory[(localr + 1):(localr + 1)+terminal, self.meta['score']]
        rewards = score_post - score_pre

        if self.clip:
            rewards = np.clip(rewards, -self.clip, self.clip)
        else:
            rewards = rewards / self.meta['avg_score']

        if trajectory[localr + terminal, self.meta['terminal']]:
            rewards[-1] = self.termination_reward

        discounts = self.discount ** np.arange(len(rewards))
        f = np.multiply(rewards, discounts).sum()

        score = (trajectory[-1, self.meta['score']] - trajectory[localr, self.meta['score']]).astype(np.float32) / self.meta['avg_score']

        # w = trajectory[-1, self.meta['score']].astype(np.float32) / self.meta['avg_score']

        is_final = int(local == (self.meta[str(epoch)] - 3))

        return {'s': s, 'a': a, 'a_tag': a_tag, 'r': r.astype(np.float32),
                's_tag': s_tag, 't': t.astype(np.float32),
                'k': np.array([k], dtype=np.float32), 'i': index, 'f': f.astype(np.float32),
                'is_final': is_final, 'a_index': a_index, 'score': score.astype(np.float32),
                'horizon_pre': horizon_pre, 'horizon_pre_tag': horizon_pre_tag,
                'matched_option': matched_option}

    def get_episode(self, index):
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

    def preprocess_history(self, epoch, frame):
        frame0 = [os.path.join(self.meta['screens_dir'], str(epoch), "%d.png" % (frame - i * self.skip)) for i in range(self.history_length)]
        frame1 = [os.path.join(self.meta['screens_dir'], str(epoch), "%d.png" % (frame + 1 - i * self.skip)) for i in range(self.history_length)]

        imgs = [preprocess_screen((f0, f1)) for f0, f1 in zip(frame0, frame1)]

        return np.stack(imgs, axis=0)


# change behavior in the test dataset case: no prioritized replay, n_steps should follow the train dataset
class DemonstrationBatchSampler(object):

    def __init__(self, dataset, train=True):
        self.dataset = dataset
        self.train = train
        self.batch = args.batch
        self.update_memory_interval = args.update_memory_interval
        self.prioritized_replay = args.prioritized_replay
        self.balance = args.balance
        self.alternating = args.alternating

    def __iter__(self):

        n = self.dataset.n
        n_batches = int(n / self.batch)
        shuffle_batch = np.copy(self.dataset.indices)

        if not self.balance and not self.alternating and (not self.prioritized_replay or not self.train):
            for i in range(len(self)):
                if not i % n_batches:
                    np.random.shuffle(shuffle_batch)
                    indices = shuffle_batch[:n_batches * self.batch].reshape((self.batch, n_batches))
                yield indices[:, i % n_batches]
        else:
            for i in range(len(self)):
                if not i % self.update_memory_interval:

                    if self.train and int(i / self.update_memory_interval) == 0:
                        prob = self.dataset.probabilities_a
                    elif self.train and int(i / self.update_memory_interval) == 1:
                        prob = self.dataset.probabilities_b
                    else:
                        prob = self.dataset.probabilities

                    indices = np.random.choice(self.dataset.indices, size=(self.batch, self.update_memory_interval),
                                               p=prob)
                yield indices[:, i % self.update_memory_interval]

    def __len__(self):
        return int(len(self.dataset) / self.batch)

class SequentialDemonstrationSampler(torch.utils.data.sampler.Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):

        n_iteration = len(self.data_source.indices)
        for i in range(len(self)):
            yield self.data_source.indices[i % n_iteration]

    def __len__(self):
        return len(self.data_source)

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

# previous implementations:

        # get action:
        # vals, counts = np.unique(np.extract(actions > 0, actions), return_counts=True)
        # a_tag = actions[0] if not actions.max() else vals[np.argmax(counts)]

    # no gamma discount factor
    # r = trajectory[local + k, self.meta['score']] - trajectory[local, self.meta['score']]
