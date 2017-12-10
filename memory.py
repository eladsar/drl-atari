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


def preprocess():
    if os.path.isdir(os.path.join(args.indir, 'data')):
        if os.path.isfile(
                os.path.join(args.indir, 'data', "%s_data.npz" % args.game)) and not args.reload_data:
            data = np.load(os.path.join(args.indir, 'data', "%s_data.npz" % args.game))
            meta = np.load(os.path.join(args.indir, 'data', "%s_meta.npz" % args.game))

            meta = {k: meta[k] for k in meta}
            data = {k: data[k] for k in data}

            meta['screens_dir'] = str(meta['screens_dir'])
            meta['trajectories_dir'] = str(meta['trajectories_dir'])

            return data, meta
    else:
        os.mkdir(os.path.join(args.indir, 'data'))

    logger.info("Parse trajectories into dict")

    pattern = "{:d},{:d},{:d},{:yesno},{:d}"
    pattern_dict = dict(yesno=lambda x: x is 'True')
    data = {}
    flat = []
    meta = {'n_states': 0, 'n_epochs': 0,
            'trajectories_dir': os.path.join(args.indir, 'trajectories', args.game),
            'screens_dir': os.path.join(args.indir, 'screens', args.game),
            'frame': 0, 'reward': 1, 'score': 2, 'terminal': 3, 'action': 4, 'terminal_states': []}

    for file in tqdm(os.listdir(meta['trajectories_dir'])):
        with open(os.path.join(meta['trajectories_dir'], file)) as fo:
            trajectory = fo.read(-1)

        # two first lines are meta data and parameters names
        trajectory = trajectory.split("\n")[2:]
        trajectory = np.array([parse.parse(pattern, i, pattern_dict).fixed for i in trajectory if i is not ""],
                              dtype=np.int)

        trajectory_skip = trajectory[::args.skip, :]
        epoch = file.split(".")[0]
        meta[epoch] = trajectory_skip.shape[0]
        meta['n_states'] += meta[epoch]
        meta['n_epochs'] += 1
        meta['terminal_states'].append(meta['n_states'] - 1)

        # concatenate actions of skip number consecutive steps
        actions = trajectory[:, meta['action']]

        # A a hack to get the skip - mod and the correct inverse sign quotient
        m = np.mod(-actions.size, args.skip)
        q, _ = np.divmod(actions.size, args.skip)

        actions = np.concatenate((actions, np.zeros(m, dtype=np.int)))
        actions = actions.reshape((q + int(m > 0), args.skip))
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

    return data, meta


def divide_dataset(meta):
    # set train and test datasets
    n = meta['n_states']
    indexes = range(n)

    # filtered_states = [i + meta['terminal_states'] for i in range(args.history_length + 1)]
    filtered_states = [i + range(args.history_length) for i in meta['terminal_states']]
    filtered_states = np.concatenate((np.arange(args.history_length - 1), *filtered_states))
    # filter terminal states and init states with no history
    indexes = [i for i in indexes if i not in filtered_states]

    p = args.test_percentage
    set_type = np.random.choice([0, 1], size=n, p=[1 - p, p])

    meta['train'] = np.array([i for i in indexes if not set_type[i]])
    meta['test'] = np.array([i for i in indexes if set_type[i]])

    return meta




class Memory(torch.utils.data.Dataset):

    def __init__(self):
        super(Memory, self).__init__()

    def __len__(self):
        return args.n_tot

class DemonstrationMemory(Memory):

    data, meta = preprocess()
    meta = divide_dataset(meta)

    def __init__(self, name):
        super(DemonstrationMemory, self).__init__()
        self.clip = np.array([args.clip if args.clip > 0 else np.inf])
        self.actions_transform = np.array(consts.action2activation[args.game])

        self.indices = DemonstrationMemory.meta[name]
        self.n = self.indices.size
        self.probabilities = np.ones(self.n) / self.n

        # self.indices = torch.from_numpy(DemonstrationMemory.meta[name])
        # self.n = self.indices.size()[0]
        # self.probabilities = torch.from_numpy(np.ones(self.n) / self.n)

    def __getitem__(self, index):

        # index = self.indices[index]
        # try:
        #     index = self.indices[index]
        # except:
        #     print("XXX")

        # index = np.random.choice(self.indices, p=self.probabilities)
        # index = 1
        epoch, frame, local, terminal = self.meta['flat'][index]
        trajectory = self.data[str(epoch)]

        s = self.preprocess_history(epoch, frame)
        s_tag = self.preprocess_history(epoch, frame + 1)

        # get the most frequent action. Action NOOP is chosen only if all the series is NOOP
        actions = trajectory[local, self.meta['action']:self.meta['action'] + args.skip]
        vals, counts = np.unique(np.extract(actions > 0, actions), return_counts=True)
        a = actions[0] if not actions.max() else vals[np.argmax(counts)]
        # a = trajectory[local, self.meta['action']]

        # transform a to a valid activation
        a = self.actions_transform[a]

        r = trajectory[local + 1, self.meta['score']] - trajectory[local, self.meta['score']]
        r = r / self.meta['avg_score']
        # r = np.max([min(r, self.clip), -self.clip])

        return {'s': s, 'a': a,
                'r': r,
                's_tag': s_tag, 't': trajectory[local + 1, self.meta['terminal']]}

    def preprocess_history(self, epoch, frame):
        files = [os.path.join(self.meta['screens_dir'], str(epoch), "%d.png" % (frame - i)) for i in range(args.history_length)]
        imgs = [preprocess_screen(file) for file in files]
        return np.stack(imgs, axis=0)


class DemonstrationBatchSampler(object):

    def __init__(self, dataset, batch_size, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):

        for i in range(len(self)):
            # indices = torch.multinomial(self.dataset.probabilities, self.batch_size, replacement=True)
            # batch = self.dataset.indices[indices]
            if not i % args.update_memory_interval:
                indices = np.random.choice(self.dataset.indices, size=(self.batch_size, args.update_memory_interval),
                                           p=self.dataset.probabilities)

            yield indices[:, i % args.update_memory_interval]

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size




