import cv2
import csv
import torch
import numpy as np
import kornia as K

from cv2 import Mat
from collections import namedtuple

# Helper functions

# named tuple to hold: Kalibration Rotation Translation
Gt = namedtuple('Gt', ['K', 'R', 'T'])

# A very small number
eps = 1e-15


def get_tensor_from_np(img: Mat, device, re_scale=840) -> (torch.Tensor, float, float):
    """ given image and rescale, calculates scale factor. resizing and converting to tensor.
    returns the tensor together with the scales to transpose back to original scale. """
    scale = re_scale / max(img.shape[0], img.shape[1])
    h = int(img.shape[0] * scale)
    w = int(img.shape[1] * scale)
    scale_h = h / img.shape[0]
    scale_w = w / img.shape[1]
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, keepdim=False).float() / 255
    # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.

    # cv2 imread, converted image to BGR. here we transform to RGB.
    # img = K.color.bgr_to_rgb(img)
    return img.to(device), scale_h, scale_w


def get_loftr_matches(img1, img2, matcher):
    """ Given img1, img2 and matcher (which is LoFTR torch.model)
    returns the matching key points 0, 1, and confidence by LoFTR. """
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }
    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0']
    mkpts1 = correspondences['keypoints1']
    conf = correspondences['confidence']

    return mkpts0, mkpts1, conf


# Given functions doing hardcore linear algebra

def NormalizeKeypoints(keypoints, K):
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints


def ComputeEssentialMatrix(F, K1, K2, kp1, kp2):
    '''Compute the Essential matrix from the Fundamental matrix, given the calibration matrices.
    Note that we ask participants to estimate F, i.e., without relying on known intrinsics.'''

    # Some shit about old opencv ransac versions returns Y x 3 matrices where Y != 3
    assert F.shape[0] == 3, 'Malformed F?'

    # Use OpenCV's recoverPose to solve the cheirality check:
    # https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)

    kp1n = NormalizeKeypoints(kp1, K1)
    kp2n = NormalizeKeypoints(kp2, K2)
    # Probably using SVD to decompose R and T.
    num_inliers, R, T, mask = cv2.recoverPose(E, kp1n, kp2n)

    return E, R, T


def QuaternionFromMatrix(matrix):
    '''Transform a rotation matrix into a quaternion.'''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array(
        [[m00 - m11 - m22, 0.0, 0.0, 0.0],
         [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
         [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
         [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]]
    )
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    '''Compute the error metric for a single example.

    The function returns two errors, over rotation and translation.
    These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.'''

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t


def LoadCalibration(filename):
    '''Load calibration data (ground truth) from the csv file.'''

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[1]
            K = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[3].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[4].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict


def ReadCovisibilityData(filename):
    covisibility_dict = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            covisibility_dict[row[1]] = float(row[2])  # the 1st column is the df index

    return covisibility_dict


def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)
