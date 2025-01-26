import os
import sys
import cv2
import csv

import torch
import random
import asyncio
import numpy as np
import kornia.feature as KF

from tqdm import tqdm
from tqdm.asyncio import tqdm as a_tqdm

from cvi.end_project.src.utils.utils import *


async def compute_error(scene, mkpts0, mkpts1, F, inliers, id1, id2, calib_dict, scaling_dict):
    inlier_kp_1 = mkpts0[inliers]
    inlier_kp_2 = mkpts1[inliers]

    # Compute the essential matrix.
    E, R, T = await asyncio.to_thread(
        ComputeEssentialMatrix, F, calib_dict[id1].K, calib_dict[id2].K, inlier_kp_1, inlier_kp_2)
    q = await asyncio.to_thread(QuaternionFromMatrix, R)
    T = T.flatten()

    # Get the relative rotation and translation between these two cameras, given their R and T in the global reference frame.
    R1_gt, T1_gt = calib_dict[id1].R, calib_dict[id1].T.reshape((3, 1))
    R2_gt, T2_gt = calib_dict[id2].R, calib_dict[id2].T.reshape((3, 1))
    dR_gt = np.dot(R2_gt, R1_gt.T)
    dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
    q_gt = await asyncio.to_thread(QuaternionFromMatrix, dR_gt)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

    # Compute the error for this example.
    err_q, err_t = await asyncio.to_thread(ComputeErrorForOneExample, q_gt, dT_gt, q, T, scaling_dict[scene])
    return err_q, err_t


async def calculate_f_and_error(scene, pair, semaphore, print_lock, calib_dict, images_dict, loftr_scale, scaling_dict,
                                matcher):
    async with semaphore:
        id1, id2 = pair.split('-')

        image_1, img1_h_scale, img1_w_scale = await asyncio.to_thread(
            get_tensor_from_np, images_dict[id1], device, loftr_scale)
        image_2, img2_h_scale, img2_w_scale = await asyncio.to_thread(
            get_tensor_from_np, images_dict[id2], device, loftr_scale)

        mkpts0, mkpts1, conf = get_loftr_matches(image_1, image_2, matcher)

        # transforming points to original scale
        mkpts0[:, 0] /= img1_h_scale
        mkpts0[:, 1] /= img1_w_scale
        mkpts1[:, 0] /= img2_h_scale
        mkpts1[:, 1] /= img2_w_scale

        mask = conf > 0.1
        mkpts0 = mkpts0[mask].cpu().numpy()
        mkpts1 = mkpts1[mask].cpu().numpy()

        F, inliers = await asyncio.to_thread(cv2.findFundamentalMat, mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.9999, 100000)

        # Compute the error for this example.
        err_q, err_t = await compute_error(scene, mkpts0, mkpts1, F, inliers, id1, id2, calib_dict, scaling_dict)

        async with print_lock:
            print(f'{pair}, err_q={(err_q):.02f} (deg), err_t={(err_t):.02f} (m)', flush=True)

        return {
            "scene": scene,
            "pair": pair,
            "err_q": err_q,
            "err_t": err_t
        }


async def fk_my_life(device, print_lock, semaphore, matcher, loftr_scale, max_pairs_per_scene):
    data_src = '../../data/cv-22928-2025-a-project/train/'
    scaling_dict = {}
    with open(f'{data_src}/scaling_factors.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            scaling_dict[row[1]] = float(row[2])

    # We use two different sets of thresholds over rotation and translation.
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)

    # Save the per-sample errors and the accumulated metric to dictionaries, for later inspection.
    errors = {scene: {} for scene in scaling_dict.keys()}
    mAA = {scene: {} for scene in scaling_dict.keys()}

    async def calculate_mAA():
        for scene in scaling_dict.keys():
            # Load ground truth data.
            calib_dict = LoadCalibration(f'{data_src}/{scene}/calibration.csv')
            # Load all pairs, find those with a co-visibility over 0.1, and subsample them.
            covisibility_dict = ReadCovisibilityData(f'{data_src}/{scene}/pair_covisibility.csv')
            pairs = [pair for pair, covis in covisibility_dict.items() if covis >= 0.5]

            print(
                f'-- Processing scene "{scene}": found {len(pairs)} pairs (will keep {min(len(pairs), max_pairs_per_scene)})',
                flush=True)

            # Subsample the pairs. Note that they are roughly sorted by difficulty (easy ones first),
            # so we shuffle them beforehand: results would be misleading otherwise.
            random.shuffle(pairs)
            pairs = pairs[:max_pairs_per_scene]

            ids = []
            for pair in pairs:
                cur_ids = pair.split('-')
                ids += cur_ids
            ids = list(set(ids))

            images_dict = {}

            for id in tqdm(ids):
                images_dict[id] = cv2.cvtColor(cv2.imread(f'{data_src}/{scene}/images/{id}.jpg'), cv2.COLOR_BGR2RGB)

            max_err_acc_q_new = []
            max_err_acc_t_new = []

            tasks = []
            for counter, pair in enumerate(pairs):
                tasks.append(
                    calculate_f_and_error(scene, pair, semaphore, print_lock, calib_dict, images_dict, loftr_scale,
                                          scaling_dict, matcher))

            results = await asyncio.gather(*tasks)

            for r in results:
                errors[r['scene']][r['pair']] = r['err_q'], r['err_t']

            # Histogram the errors over this scene.
            mAA[scene] = ComputeMaa([v[0] for v in errors[scene].values()], [v[1] for v in errors[scene].values()],
                                    thresholds_q, thresholds_t)
            print()
            print(f'Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
            print()
        return mAA

    mAA = await calculate_mAA()

    print()
    print('------- SUMMARY -------')
    print()
    for scene in scaling_dict.keys():
        print(f'-- Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
    print()
    print(f'Mean average Accuracy on dataset: {np.mean([mAA[scene][0] for scene in mAA]):.05f}')

    pass


if __name__ == '__main__':
    random.seed(1337)

    device = torch.device('cuda')
    matcher = KF.LoFTR()
    matcher = matcher.to(device).eval()

    kwargs = {
        "device": device,
        "matcher": matcher,
        "print_lock": asyncio.Lock(),
        "semaphore": asyncio.Semaphore(10),
        "loftr_scale": 1080,
        "max_pairs_per_scene": 25
    }

    asyncio.run(fk_my_life(**kwargs))
