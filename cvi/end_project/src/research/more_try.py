import sys, os
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
import math
import time
from sklearn.cluster import KMeans
from PIL import Image
from scipy.spatial import cKDTree
#import pvsac

sys.path.append('/kaggle/input/imc2022-dependencies/DKM/')

dry_run = False

################### Model ##########################################################

device = torch.device('cuda')
matcher = KF.LoFTR(pretrained='outdoor')
#matcher.load_state_dict(torch.load("/kaggle/input/kornia-loftr/loftr_outdoor.ckpt")['state_dict'])
matcher = matcher.to(device).eval()

#torch.hub.set_dir('/kaggle/working/pretrained/')
#from dkm import dkm_base

#model = dkm_base(pretrained=True, version="v11")

src = '../../data/cv-22928-2025-a-project/'
test_samples = []

with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


#####################################################################################
def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''

    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def get_images(image, center_point, crop_size=512):
    h, w, c = image.shape
    cw = int(w / 2)
    ch = int(h / 2)
    cx = int(center_point[0])
    cy = int(center_point[1])

    half_crop = crop_size // 2

    if ch > cy:
        ylb = max(0, cy - half_crop)
        yub = min(h, max(0, cy - half_crop) + crop_size)
    else:
        ylb = max(0, min(h, cy + half_crop) - crop_size)
        yub = min(h, cy + half_crop)
    if cw > cx:
        xlb = max(0, cx - half_crop)
        xub = min(w, max(0, cx - half_crop) + crop_size)
    else:
        xlb = max(0, min(w, cx + half_crop) - crop_size)
        xub = min(w, cx + half_crop)
    d_img = image[ylb:yub, xlb:xub]

    return d_img, [w, h, cx, cy, xlb, xub, ylb, yub]


def get_matches(image_1, image_2):
    input_dict = {"image0": K.color.rgb_to_grayscale(image_1),
                  "image1": K.color.rgb_to_grayscale(image_2)}
    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()
    return mkpts0, mkpts1, conf


def concatenate_matches(mkpts0, mkpts1, score, d_score, d_mkpts0, d_mkpts1, info_1, info_2):
    s_mkpts0 = (d_mkpts0 + [info_1[4], info_1[6]]).astype(float)
    s_mkpts1 = (d_mkpts1 + [info_2[4], info_2[6]]).astype(float)

    f_mkpts0 = np.concatenate((mkpts0, s_mkpts0))
    f_mkpts1 = np.concatenate((mkpts1, s_mkpts1))
    f_score = np.concatenate((score, d_score))

    return f_mkpts0, f_mkpts1, f_score


def set_torch_image(img, device, infer_size=840):
    scale = infer_size / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    scale_w = w / img.shape[1]
    scale_h = h / img.shape[0]
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device), (scale_h, scale_w)


def GetLargestNumbers(arr):
    unique_nums = set(arr)
    sorted_nums = sorted(unique_nums, reverse=True)
    largestIndex = np.where(arr == sorted_nums[0])[0][0]
    secondIndex = np.where(arr == sorted_nums[1])[0][0]
    return largestIndex, secondIndex


def GetMostDenseCenterPoint(n_clusters, mkpts0, mkpts1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(mkpts0)
    most_dense_center_points = []
    second_dense_center_points = []
    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)
    largest_count, second_count = GetLargestNumbers(counts)
    max_counts_label = unique[largest_count]
    second_counts_label = unique[second_count]

    dense_mkpts0 = []
    dense_mkpts1 = []
    indices = np.where(labels == max_counts_label)
    for matchIndex in indices:
        dense_mkpts0.append(mkpts0[matchIndex])
        dense_mkpts1.append(mkpts1[matchIndex])
    most_dense_center_points.append(np.array([np.mean(dense_mkpts0[0][:, 0]), np.mean(dense_mkpts0[0][:, 1])]))
    most_dense_center_points.append(np.array([np.mean(dense_mkpts1[0][:, 0]), np.mean(dense_mkpts1[0][:, 1])]))

    std1 = np.std(np.linalg.norm(dense_mkpts0[0] - np.expand_dims(most_dense_center_points[0], axis=0), axis=1))
    std2 = np.std(np.linalg.norm(dense_mkpts1[0] - np.expand_dims(most_dense_center_points[1], axis=0), axis=1))

    second_mkpts0 = []
    second_mkpts1 = []
    indices2 = np.where(labels == second_counts_label)
    for matchIndex2 in indices2:
        second_mkpts0.append(mkpts0[matchIndex2])
        second_mkpts1.append(mkpts1[matchIndex2])
    second_dense_center_points.append(np.array([np.mean(second_mkpts0[0][:, 0]), np.mean(second_mkpts0[0][:, 1])]))
    second_dense_center_points.append(np.array([np.mean(second_mkpts1[0][:, 0]), np.mean(second_mkpts1[0][:, 1])]))

    return most_dense_center_points, second_dense_center_points, (std1, std2)


def find_Homography(mkpts0, mkpts1, img1_o):
    Hmatrix, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.99999, 10000)
    h, w, c = img1_o.shape
    if not homography_is_accepted(Hmatrix):
        return None, None
    img1_w = cv2.warpPerspective(img1_o, Hmatrix, (w, h))
    return Hmatrix, img1_w


def GetLoFTRMatches(img1_o, img2_o, device, infer_sizes):
    img1_tensor, (scale_h1, scale_w1) = set_torch_image(img1_o, device, infer_sizes[0])
    img2_tensor, (scale_h2, scale_w2) = set_torch_image(img2_o, device, infer_sizes[1])
    mkpts0, mkpts1, conf = get_matches(img1_tensor, img2_tensor)
    mkpts0[:, 0] /= scale_w1
    mkpts0[:, 1] /= scale_h1
    mkpts1[:, 0] /= scale_w2
    mkpts1[:, 1] /= scale_h2

    return mkpts0, mkpts1, conf


# def GetDkmMatches(crop_img1, crop_img2, infer_sizes=(512, 512), num_mkpts=500):
#     h1, w1, _ = crop_img1.shape
#     scale1 = infer_sizes[0] / max(h1, w1)
#     new_h1, new_w1 = int(h1 * scale1), int(w1 * scale1)
#     crop_img1 = cv2.resize(crop_img1, (new_w1, new_h1))
#     scale_h1, scale_w1 = new_h1 / h1, new_w1 / w1
#
#     h2, w2, _ = crop_img2.shape
#     scale2 = infer_sizes[1] / max(h2, w2)
#     new_h2, new_w2 = int(h2 * scale2), int(w2 * scale2)
#     crop_img2 = cv2.resize(crop_img2, (new_w2, new_h2))
#     scale_h2, scale_w2 = new_h2 / h2, new_w2 / w2
#
#     crop_img1_PIL = Image.fromarray(cv2.cvtColor(crop_img1, cv2.COLOR_BGR2RGB))
#     crop_img2_PIL = Image.fromarray(cv2.cvtColor(crop_img2, cv2.COLOR_BGR2RGB))
#     dense_matches, dense_certainty = model.match(crop_img1_PIL, crop_img2_PIL)
#     sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, num_mkpts)
#     dkm_crop_mkpts0 = sparse_matches[:, :2]
#     dkm_crop_mkpts1 = sparse_matches[:, 2:]
#
#     h, w, c = crop_img1.shape
#     dkm_crop_mkpts0[:, 0] = (dkm_crop_mkpts0[:, 0] + 1) / 2 * w / scale_w1
#     dkm_crop_mkpts0[:, 1] = (dkm_crop_mkpts0[:, 1] + 1) / 2 * h / scale_h1
#
#     h, w, c = crop_img2.shape
#     dkm_crop_mkpts1[:, 0] = (dkm_crop_mkpts1[:, 0] + 1) / 2 * w / scale_w2
#     dkm_crop_mkpts1[:, 1] = (dkm_crop_mkpts1[:, 1] + 1) / 2 * h / scale_h2
#
#     return dkm_crop_mkpts0, dkm_crop_mkpts1, sparse_certainty


def GetWarpedLoFTRMatches(mkpts0, mkpts1, score, img1_w, img2_o, corrected_Hmatrix, device, infer_sizes=(840, 840)):
    if corrected_Hmatrix is not None:
        corrected_Hmatrix_inv = np.linalg.inv(corrected_Hmatrix)
        mkpts0_w, mkpts1_w, score_w = GetLoFTRMatches(img1_w, img2_o, device, infer_sizes)

        if mkpts0_w.shape[0] >= 8:
            mkpts0_w = np.expand_dims(mkpts0_w, axis=1)
            mkpts0_w = cv2.perspectiveTransform(mkpts0_w, corrected_Hmatrix_inv).squeeze()
            mkpts0 = np.concatenate([mkpts0, mkpts0_w])
            mkpts1 = np.concatenate([mkpts1, mkpts1_w])
            score = np.concatenate([score, score_w])
            return mkpts0, mkpts1, score

    return mkpts0, mkpts1, score


def radius_NMS(kps, kps1, score, r=0.3):
    if len(kps) == 0:
        return kps, kps1

    order = np.argsort(score)[::-1]
    kps = np.array(kps)[order]
    kps1 = np.array(kps1)[order]
    data_pts = kps

    kd_tree = cKDTree(data_pts)
    N = len(kps)
    idxs_removed = set()

    kd_idxs = kd_tree.query_ball_point(data_pts, r)

    for i in range(N):
        if i in idxs_removed:
            continue
        for j in kd_idxs[i]:
            if j > i:
                idxs_removed.add(j)
    idxs_remaining = [i for i in range(N) if i not in idxs_removed]

    kps_out = kps[idxs_remaining]
    kps1_out = kps1[idxs_remaining]
    return kps_out, kps1_out


def homography_is_accepted(H):
    H /= H[2, 2]
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if det < 0:
        return False
    N1 = math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
    N2 = math.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)

    if N1 > 100 or N1 < 0.001:
        return False
    if N2 > 100 or N2 < 0.001:
        return False
    return True


def cut_mkpts(f_mkpts0, f_mkpts1, f_score, img1_o, img2_o):
    maskx_0 = (f_mkpts0[:, 0] >= 0) & (f_mkpts0[:, 0] <= img1_o.shape[1] - 1)
    f_mkpts0, f_mkpts1, f_score = f_mkpts0[maskx_0, :], f_mkpts1[maskx_0, :], f_score[maskx_0]
    masky_0 = (f_mkpts0[:, 1] >= 0) & (f_mkpts0[:, 1] <= img1_o.shape[0] - 1)
    f_mkpts0, f_mkpts1, f_score = f_mkpts0[masky_0, :], f_mkpts1[masky_0, :], f_score[masky_0]

    maskx_1 = (f_mkpts1[:, 0] >= 0) & (f_mkpts1[:, 0] <= img2_o.shape[1] - 1)
    f_mkpts0, f_mkpts1, f_score = f_mkpts0[maskx_1, :], f_mkpts1[maskx_1, :], f_score[maskx_1]
    masky_1 = (f_mkpts1[:, 1] >= 0) & (f_mkpts1[:, 1] <= img2_o.shape[0] - 1)
    f_mkpts0, f_mkpts1, f_score = f_mkpts0[masky_1, :], f_mkpts1[masky_1, :], f_score[masky_1]
    return f_mkpts0, f_mkpts1, f_score


####################

F_dict = {}

for i, row in enumerate(test_samples):
    np.random.seed(42)
    sample_id, batch_id, img1_id, img2_id = row

    # Load the images.
    st = time.time()
    img1_o = cv2.imread(f'{src}/test_images/{batch_id}/{img1_id}.jpg')
    img2_o = cv2.imread(f'{src}/test_images/{batch_id}/{img2_id}.jpg')
    h1, w1, c1 = img1_o.shape
    h2, w2, c2 = img2_o.shape

    default_infer_sizes = (1080, 1080)
    crop_sizes = (512, 512)

    ##################### GET MATCH POINTS FROM WHOLE IAMGES ############################
    mkpts0, mkpts1, score = GetLoFTRMatches(img1_o, img2_o, device, infer_sizes=default_infer_sizes)

    ##################### GET CROP POINTS FROM MATCH POINTS  ############################
    n_clusters = 4
    first_center_points, second_center_points, (std1, std2) = GetMostDenseCenterPoint(n_clusters, mkpts0, mkpts1)

    ##################### GET MATCH POINTS FROM WARPING IMAGE ###########################
    if mkpts0.shape[0] > 5:
        corrected_Hmatrix1, img1_w = find_Homography(mkpts0, mkpts1, img1_o)
        corrected_Hmatrix2, img2_w = find_Homography(mkpts1, mkpts0, img2_o)
        mkpts0, mkpts1, score = GetWarpedLoFTRMatches(mkpts0, mkpts1, score, img1_w, img2_o, corrected_Hmatrix1, device,
                                                      infer_sizes=(1080, 1080))
        mkpts1, mkpts0, score = GetWarpedLoFTRMatches(mkpts1, mkpts0, score, img2_w, img1_o, corrected_Hmatrix2, device,
                                                      infer_sizes=(1080, 1080))

    ##################### GET MATCH POINTS FROM CROP IMAGE ##############################
    if max(h1, w1) >= 512:
        crop_img1, info_1 = get_images(img1_o, first_center_points[0],
                                       crop_size=crop_sizes[0])  # [w, h, cx, cy, xlb, xub, ylb, yub]
        crop_img2, info_2 = get_images(img2_o, first_center_points[1], crop_size=crop_sizes[1])

        # Get LoFTR matching points
        loftr_crop_mkpts0, loftr_crop_mkpts1, loftr_score = GetLoFTRMatches(crop_img1, crop_img2, device,
                                                                            infer_sizes=(1080, 1080))

        if loftr_crop_mkpts0.shape[0] < 500 and loftr_crop_mkpts0.shape[0] > 5:
            corrected_Hmatrix_crop1, crop_img1_w = find_Homography(loftr_crop_mkpts0, loftr_crop_mkpts1, crop_img1)
            corrected_Hmatrix_crop2, crop_img2_w = find_Homography(loftr_crop_mkpts1, loftr_crop_mkpts0, crop_img2)
            loftr_crop_mkpts0, loftr_crop_mkpts1, loftr_score = GetWarpedLoFTRMatches(loftr_crop_mkpts0,
                                                                                      loftr_crop_mkpts1, loftr_score,
                                                                                      crop_img1_w, crop_img2,
                                                                                      corrected_Hmatrix_crop1,
                                                                                      device, infer_sizes=(840, 840))
            loftr_crop_mkpts1, loftr_crop_mkpts0, loftr_score = GetWarpedLoFTRMatches(loftr_crop_mkpts1,
                                                                                      loftr_crop_mkpts0, loftr_score,
                                                                                      crop_img2_w, crop_img1,
                                                                                      corrected_Hmatrix_crop2,
                                                                                      device, infer_sizes=(840, 840))

        # # Get DKM matching points
        # dkm_crop_mkpts0, dkm_crop_mkpts1, dkm_score = GetDkmMatches(crop_img1, crop_img2, infer_sizes=(1080, 1080),
        #                                                             num_mkpts=300)
        #
        # # concatenate matches
        # mkpts0, mkpts1, score = concatenate_matches(mkpts0, mkpts1, score, loftr_score, loftr_crop_mkpts0,
        #                                             loftr_crop_mkpts1, info_1, info_2)
        # f_mkpts0, f_mkpts1, f_score = concatenate_matches(mkpts0, mkpts1, score, dkm_score, dkm_crop_mkpts0,
        #                                                   dkm_crop_mkpts1, info_1, info_2)

    else:
        f_mkpts0, f_mkpts1 = mkpts0, mkpts1

    ##################### FILTERING MATCH POINTS#### ####################################
    def estimate_fundamental_matrix(f_mkpts0, f_mkpts1, threshold=0.1, confidence=0.99999, max_iters=100000):
        # Use cv2.findFundamentalMat with USAC_MAGSAC for robust estimation
        F, inliers = cv2.findFundamentalMat(
            f_mkpts0, f_mkpts1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=threshold,
            confidence=confidence,
            maxIters=max_iters
        )
        return F, inliers

    f_mkpts0, f_mkpts1, f_score = cut_mkpts(mkpts0, mkpts1, score, img1_o, img2_o)
    f_mkpts0, f_mkpts1 = radius_NMS(f_mkpts0, f_mkpts1, f_score, r=0.5)
    if len(f_mkpts0) > 7:
        F, inliers = estimate_fundamental_matrix(f_mkpts0, f_mkpts1)

        inliers = inliers > 0
        assert F.shape == (3, 3), 'Malformed F?'
        F_dict[sample_id] = F
    else:
        F_dict[sample_id] = np.zeros((3, 3))
        continue

    gc.collect()
    nd = time.time()

    ##################### SAVE DATA FOR SAMPLE DATA VISUALIZATION #######################
    if (i < 3):
        print("Running time: ", nd - st, " s")
        np.save(f'../../data/f_mkpts0_{i}', f_mkpts0)
        np.save(f'../../data/f_mkpts1_{i}', f_mkpts1)
        np.save(f'../../data/img1_o_{i}', img1_o)
        np.save(f'../../data/img2_o_{i}', img2_o)
        np.save(f'../../data/inliers_{i}', inliers)

with open('./submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')