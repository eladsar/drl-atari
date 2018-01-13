import numpy as np
import torch
import os
import parse
from tqdm import tqdm
import cv2

from config import args, consts

# local consts

img_width = args.width
img_height = args.height

# consts:

# cv2.NORM_L2: 4
# cv2.CV_32F: 5
# cv2.COLOR_BGR2GRAY

img_dtype = cv2.CV_32F
img_norm = cv2.NORM_MINMAX
img_bgr2gray = cv2.COLOR_BGR2GRAY
img_rgb2gray = cv2.COLOR_RGB2GRAY
img_gray2rgb = cv2.COLOR_GRAY2RGB
img_bgr2rgb = cv2.COLOR_BGR2RGB

img_threshold = cv2.THRESH_BINARY
img_inter = cv2.INTER_NEAREST
flicker = consts.flicker[args.game]

def convert_screen_to_rgb(img, resize=False):
    img = cv2.cvtColor(img.numpy(), img_gray2rgb)
    #
    if resize:
        img = img / img.max()
        img = cv2.resize(img, (128, 1024), interpolation=img_inter)
    return torch.from_numpy(np.rollaxis(img, 2, 0))

def preprocess_screen(imgs):

    if type(imgs[0]) is str:
        img0 = cv2.cvtColor(cv2.imread(imgs[0]), img_bgr2rgb)
        img1 = cv2.cvtColor(cv2.imread(imgs[1]), img_bgr2rgb)

    else:
        img0, img1 = imgs
        # veritcal shift with gym
        # img0 = np.pad(img0[:-2], ((2, 0), (0, 0), (0, 0)), 'edge')
        # img1 = np.pad(img1[:-2], ((2, 0), (0, 0), (0, 0)), 'edge')

    img0 = img0.astype(np.float32)
    img1 = img1.astype(np.float32)

    img0 = cv2.resize(img0, (img_width, img_height))
    img1 = cv2.resize(img1, (img_width, img_height))

    img0 = cv2.cvtColor(img0, img_rgb2gray)
    img1 = cv2.cvtColor(img1, img_rgb2gray)

    # ret, img = cv2.threshold(img, 1, 1, img_threshold)

    # img = cv2.resize(img + 0.5, (img_width, img_height))
    # img = cv2.resize(img, (img_width, img_height + 26))
    # img = img[26:110, :]


    # img = img / 256.
    # img = cv2.normalize(img, None, norm_type=img_norm, dtype=img_dtype)

    # _, img0 = cv2.threshold(img0, 1, 100, img_threshold)
    # _, img1 = cv2.threshold(img1, 1, 200, img_threshold)

    if flicker:
        img = np.maximum(img0, img1)
    else:
        img = img0

    # _, img = cv2.threshold(img, 1, 1, img_threshold)
    img = img / 256.

    img += np.random.randn(84, 84) * 0.005
    img = cv2.normalize(img, None, alpha=0.001, beta=0.999, norm_type=img_norm, dtype=img_dtype)

    return img



# def preprocess_screen(img):
#
#     if type(img) is np.ndarray:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     else:
#         # # Load an color image in grayscale and scale it to [0, 1)
#         # img = cv2.imread(img, 0)
#         img0 = cv2.imread(img[0])
#         img1 = cv2.imread(img[1])
#         img = np.maximum(img0, img1)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     img = cv2.resize(img, (args.height, args.width))
#     img = cv2.normalize(img, img, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
#
#     return img.astype(np.float32)
