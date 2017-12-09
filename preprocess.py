import numpy as np
import os
import parse
from tqdm import tqdm
import cv2

from config import args, consts


def preprocess_screen(img):

    if type(img) is np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # # Load an color image in grayscale and scale it to [0, 1)
        img = cv2.imread(img, 0)

    img = cv2.resize(img, (args.height, args.width))
    cv2.normalize(img, img, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

    return img.astype(np.float32)


def get_trajectories_dict():

    if os.path.isdir(os.path.join(args.indir, 'data')):
        if os.path.isfile(os.path.join(args.indir, 'data', "%s_data.npz" % args.game)) and args.reuse_preprocessed_data:
            data = np.load(os.path.join(args.indir, 'data', "%s_data.npz" % args.game))
            meta = np.load(os.path.join(args.indir, 'data', "%s_meta.npz" % args.game))

            meta = {k: meta[k] for k in meta}
            if args.screensdir is not "":
                meta['screens_dir'] = os.path.join(args.screensdir, args.game)
            else:
                meta['screens_dir'] = str(meta['screens_dir'])
            meta['trajectories_dir'] = str(meta['trajectories_dir'])

            return {k: data[k] for k in data}, meta
    else:
        os.mkdir(os.path.join(args.indir, 'data'))

    pattern = "{:d},{:d},{:d},{:yesno},{:d}"
    pattern_dict = dict(yesno=lambda x: x is 'True')
    data = {}
    flat = []
    meta = {'n_states': 0, 'n_epochs':0,
              'trajectories_dir': os.path.join(args.indir, 'trajectories', args.game),
              'screens_dir': os.path.join(args.indir, 'screens', args.game),
            'frame':0, 'reward': 1, 'score': 2, 'terminal': 3, 'action': 4, 'terminal_states': []}

    print("Parse trajectories into dict")
    for file in tqdm(os.listdir(meta['trajectories_dir'])):
        with open(os.path.join(meta['trajectories_dir'], file)) as fo:
            trajectory = fo.read(-1)

        # two first lines are meta data and parameters names
        trajectory = trajectory.split("\n")[2:]
        trajectory = np.array([parse.parse(pattern, i, pattern_dict).fixed for i in trajectory if i is not ""], dtype=np.int)

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

    np.savez(os.path.join(args.indir, 'data', "%s_data" % args.game), **data)
    np.savez(os.path.join(args.indir, 'data', "%s_meta" % args.game), **meta)

    return data, meta
