import numpy as np
from skimage import io, transform, color
import torch
import cv2
import os

from config import consts, args
from preprocess import preprocess_screen

class DemonstrationsDataset(torch.utils.data.Dataset):
    def __init__(self, data, meta):
        super(DemonstrationsDataset, self).__init__()
        self.data = data
        self.meta = meta

        # set train and test datasets
        n = self.meta['n_states']
        indexes = range(n)

        filtered_states = [i + meta['terminal_states'] for i in range(args.history_length+1)]
        filtered_states = np.concatenate((np.arange(args.history_length), *filtered_states))
        # filter terminal states and init states with no history
        indexes = [i for i in indexes if i not in filtered_states]

        p = args.test_percentage
        set_type = np.random.choice([0, 1], size=n, p=[1 - p, p])

        self.meta['train'] = [i for i in indexes if not set_type[i]]
        self.meta['test'] = [i for i in indexes if set_type[i]]

        # get average score (temporal implementation)
        scores = []
        for index in meta['terminal_states']:
            epoch, frame, local, terminal = self.meta['flat'][index]
            trajectory = self.data[str(epoch)]
            scores.append(trajectory[local, self.meta['score']])

        self.meta['avg_score'] = np.mean(scores)
        print("average score is %d" % self.meta['avg_score'])


        self.clip = np.array([args.clip if args.clip > 0 else np.inf])

        self.actions_transform = np.array(consts.action2activation[args.game])

    def __len__(self):
        return self.meta['n_states']

    def __getitem__(self, index):
        epoch, frame, local, terminal = self.meta['flat'][index]
        trajectory = self.data[str(epoch)]

        s = self.preprocess_history(epoch, frame)
        s_tag = self.preprocess_history(epoch, frame + 1)

        # get the most frequent action. Action NOOP is chosen only if all the series is NOOP
        actions = trajectory[local, self.meta['action']:self.meta['action'] + args.skip]
        vals, counts = np.unique(np.extract(actions > 0, actions), return_counts=True)
        a = actions[0] if not actions.max() else vals[np.argmax(counts)]

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



