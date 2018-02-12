import os
import sys
import cv2
import copy
import numpy as np

from tqdm import tqdm

indir = "/data/sarafie/atari/atari_v2_release"
eol_dir = "/data/sarafie/atari/atari_v2_release/eol"
pics_dir = "/data/sarafie/atari/eolpics"

# games = ["spaceinvaders"]
games = ["revenge", "mspacman"]

rectangle = {"spaceinvaders": [(180, 200), (80, 100)],
             "mspacman": [(170, 190), (10, 40)],
             "revenge": [(12, 22), (55, 95)],
             }

lives = {"spaceinvaders": 2,
             "mspacman": 2,
             "revenge": 5,
             }


for game in games:

    print("Find EOL for game: %s" % game)
    game_indir = os.path.join(indir, "screens", game)
    game_eol_dir = os.path.join(eol_dir, game)

    pics = []
    for l in range(lives[game]):
        name = os.path.join(pics_dir, "%s.%d.png" % (game, l))
        im = cv2.imread(name)
        pics.append(im)

    a0 = rectangle[game][0][0]
    a1 = rectangle[game][0][1]
    b0 = rectangle[game][1][0]
    b1 = rectangle[game][1][1]

    for episode in tqdm(os.listdir(game_indir)):

        # episode = "11"
        stack = copy.copy(pics)
        eol_file = os.path.join(game_eol_dir, "%s.txt" % episode)
        with open(eol_file, "w") as fo:

            frame = 0
            while stack:

                # if frame == 794:
                #     print("Here")
                im = cv2.imread(os.path.join(game_indir, episode, "%d.png" % frame))
                if im is None:
                    break
                im = im[a0:a1, b0:b1, :]
                if np.array_equal(im, stack[-1]):
                    # print("Terminal frame: %d" % (frame-1))
                    fo.write("%d\n" % (frame-1))
                    stack.pop()
                frame += 1






