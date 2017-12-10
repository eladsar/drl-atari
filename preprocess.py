import numpy as np
import torch
import os
import parse
from tqdm import tqdm
import cv2

from config import args, consts


def convert_screen_to_rgb(img):
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_GRAY2RGB)
    return torch.from_numpy(np.rollaxis(img, 2, 0))

def preprocess_screen(img):

    if type(img) is np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # # Load an color image in grayscale and scale it to [0, 1)
        img = cv2.imread(img, 0)

    img = cv2.resize(img, (args.height, args.width))
    cv2.normalize(img, img, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

    return img.astype(np.float32)
