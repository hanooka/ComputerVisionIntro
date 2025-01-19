import asyncio
import json
import os
import numpy as np
import cv2
import csv
from glob import glob
import torch
import matplotlib.pyplot as plt
import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF
import gc

from kornia_moons.viz import draw_LAF_matches
from tensorflow.python.ops.signal.shape_ops import frame
from tqdm.asyncio import tqdm

device = torch.device('cuda')
matcher = KF.LoFTR()
matcher = matcher.to(device).eval()

src = '../../data/cv-22928-2025-a-project/'
output_file = 'submission.csv'
checkpoint_file = 'submission_checkpoint.json'

test_samples = []

with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''

    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def load_torch_image(fname, device):
    img = cv2.imread(fname)
    scale = 840 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)


F_dict = {}
import time

for i, row in enumerate(test_samples):
    sample_id, batch_id, image_1_id, image_2_id = row
    # Load the images.
    st = time.time()
    image_1 = load_torch_image(f'{src}/test_images/{batch_id}/{image_1_id}.jpg', device)
    image_2 = load_torch_image(f'{src}/test_images/{batch_id}/{image_2_id}.jpg', device)
    print(image_1.shape)
    input_dict = {"image0": K.color.rgb_to_grayscale(image_1),
                  "image1": K.color.rgb_to_grayscale(image_2)}

    with torch.no_grad():
        correspondences = matcher(input_dict)
        #print(correspondences)

    # Why not take only points with high confidence

    mkpts0 = correspondences['keypoints0']
    mkpts1 = correspondences['keypoints1']

    mask = correspondences['confidence'] > 0.5

    print(sum(mask))

    mkpts1 = mkpts1[mask].cpu().numpy()
    mkpts0 = mkpts0[mask].cpu().numpy()

    print(mkpts0.shape)
    print(mkpts1.shape)


    if len(mkpts0) > 7:
        F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.25, 0.99999, 100000)
        inliers = inliers > 0
        print("inliners", sum(inliers))
        assert F.shape == (3, 3), 'Malformed F?'
        F_dict[sample_id] = F
    else:
        F_dict[sample_id] = np.zeros((3, 3))
        continue
    gc.collect()
    nd = time.time()
    if (i < 3):
        print("Running time: ", nd - st, " s")
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                         torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                         torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                         torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                         torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(image_1),
            K.tensor_to_image(image_2),
            inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                       'tentative_color': None,
                       'feature_color': (0.2, 0.5, 1), 'vertical': False})
    plt.show()