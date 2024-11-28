import functools
import pickle
from collections import defaultdict
from typing import Tuple, Union, List
import sys

import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cv2.typing import MatLike
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClusterMixin

np.set_printoptions(threshold=sys.maxsize)

from sklearn.cluster import DBSCAN


def rotate_coordinates_back(points, rotation_matrix, original_shape):
    """
    Rotate points from a rotated image back to the original image's coordinate system.

    Parameters:
        points (np.ndarray): Array of points as (x, y).
        rotation_matrix (np.ndarray): Rotation matrix used to create the rotated image.
        original_shape (tuple): Original image dimensions as (height, width).

    Returns:
        np.ndarray: Array of points rotated back to the original coordinate system.
    """
    # Invert the rotation matrix
    original_h, original_w = original_shape

    rotation_matrix_inv = cv2.invertAffineTransform(rotation_matrix)
    homogeneous_coords = np.column_stack([points, np.ones(len(points))])
    transformed_coords = np.dot(rotation_matrix_inv, homogeneous_coords.T).T

    transformed_coords = np.clip(
        transformed_coords, [0, 0], [original_w - 1, original_h - 1]
    )
    return transformed_coords


def cluster_keypoints(keypoints, eps=5, min_samples=1):
    """
    Cluster keypoints using DBSCAN to merge nearby points.

    Parameters:
        keypoints (np.ndarray): Array of detected keypoints as (x, y).
        eps (float): Maximum distance between points to form a cluster.
        min_samples (int): Minimum points in a cluster.

    Returns:
        np.ndarray: Array of clustered keypoints as (x, y).
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(keypoints)
    unique_labels = set(clustering.labels_)
    # print(unique_labels)
    clustered_keypoints = []

    for label in unique_labels:
        if label == -1:  # Noise points, optionally handle separately
            continue
        cluster = keypoints[clustering.labels_ == label]
        centroid = cluster.mean(axis=0)  # Compute the cluster centroid
        clustered_keypoints.append(centroid)

    return np.clip(np.array(clustered_keypoints, dtype=int), a_min=0, a_max=np.inf)


def apply_harris_detector(img: MatLike, block_size: int = 2, ksize: int = 3,
                          k: float = 0.04, th_factor=0.02) -> (MatLike, np.array):
    """
    :param img: the given image, with no assumptions.
    :param block_size: size of neighborhood considered for corner detection.
    for example, 2-3 will be considered small and will be sensetive to small corners, which in terms will be
    more prune to noise.
    5-7 will define a larger window, which focus on more prominent corners, which might miss some corners.
    :param ksize: k size, define the size / kernel of the sobel operator. (3 for 3x3, 7 for 7x7 etc.)
    again, small parameter to detect sensitive/sharp changes in image which might be pruned to noise
    larger less prune to noise but might miss finer details.
    :param k: a variable harris discovered helps him make this shit work, usually set between 0.04 to 0.06.
    :param th_factor: helps determine the threshold in which we'll consider the "discovered corner" a corner.
    :return: marked/colored image with corners, keypoints
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # using harris detector implemented by opencv, to detect corners
    corners_resp = cv2.cornerHarris(gray_image, blockSize=block_size, ksize=ksize, k=k)
    threshold = th_factor * corners_resp.max()
    ret_img = img.copy()
    # mark red on copy of original image
    key_points = np.argwhere(corners_resp > threshold)
    ret_img[corners_resp > threshold] = [0, 0, 255]
    return ret_img, key_points


def apply_fast_detector(
        img,
        th: int = 10, non_max_supr: bool = True,
        neighbors_type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
) -> (MatLike, Tuple[cv2.KeyPoint]):
    """
    :param img:
    :param th: threshold helps for corner detector. a higher threshold will result in less corners,
    as the algo become stricter.
    :param non_max_supr: boolean that helps eliminate weaker key points that are close to stronger ones.
    if True, it ensures that the output contains distinct well-defined corners.
    :param neighbors_type: defines the pixel circle size used for detection
    :return:
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector()
    fast = fast.create(threshold=th, nonmaxSuppression=non_max_supr, type=neighbors_type)
    kp = fast.detect(gray_image)
    output_image = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return output_image, kp


def apply_orb_detector(img: MatLike, max_keypoints: int = 500):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB().create(nfeatures=max_keypoints)
    kp, descriptors = orb.detectAndCompute(gray_image, None)
    output_image = cv2.drawKeypoints(
        img,
        kp,
        None,
        color=(0, 255, 0),  # Green color for keypoints
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return output_image, kp, descriptors


def apply_sift_detector(
        img: MatLike,
        nfeatures: int = 0,
        nOctaveLayers: int = 3,
        contrastThreshold: float = 0.04,
        edgeThreshold: int = 10,
        sigma: float = 1.6):
    """
    Not working for some reason, I have a c++ "unknown exception" which I didn't dwelve to solve
    histogram of gradient in image. => Robust to overall change. fully affine invariant.
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT()
    sift.create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers,
                contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)

    # Detect keypoints and compute descriptors
    keypoints = sift.detect(gray_image, None)

    # Draw the detected keypoints on the image
    output_image = cv2.drawKeypoints(gray_image, keypoints, img)
    return output_image


def apply_akaze_detector(
        img,
        descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE,
        descriptor_size: int = 0,
        descriptor_channels: int = 1,
        th: float = 0.001,
        n_octaves: int = 4,
        n_octaves_layers: int = 4
):
    """
    :param img:
    :param descriptor_type:
        cv2.AKAZE_DESCRIPTOR_KAZE: Uses the original KAZE descriptor, which is based on nonlinear scale space and offers good robustness in texture-rich and noise-prone images.
        cv2.AKAZE_DESCRIPTOR_UPRIGHT: Uses the upright descriptor, where the keypoints are not oriented, which can be slightly faster but less robust to image rotation.
    :param descriptor_size:
        0 value automatically being coverted to 64 as it's opencv default. also used in SIFT and SURF algos.
    :param th:
    :param n_octaves:
    :param n_octaves_layers:
    :return:
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE().create(descriptor_type, descriptor_size, descriptor_channels, th, n_octaves, n_octaves_layers)
    keypoints, descriptors = akaze.detectAndCompute(gray_image, None)
    output_image = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_image, keypoints, descriptors


def rescale_image(img: MatLike, scale) -> (MatLike, None):
    """ Rescales image by scale factor. """
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


def rotate_image(img: MatLike, angle: Union[float, int], scale: float = 1.0) -> (MatLike, np.ndarray):
    """ Rotating the image, with angle and re-scale factor.
    considering rotated image will create new height/width. """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int(h * sin_theta + w * cos_theta)
    new_height = int(h * cos_theta + w * sin_theta)
    rotation_matrix[0, 2] += (new_width - w) / 2
    rotation_matrix[1, 2] += (new_height - w) / 2
    return cv2.warpAffine(img, rotation_matrix, (new_width, new_height)), rotation_matrix


def add_gaussian_noise(img: MatLike, mean=0, std_dev=2) -> (MatLike, None):
    """ Add gaussian noise using normal distribution. """
    noise = np.random.normal(mean, std_dev, img.shape).astype(np.float16)
    noisy_image = img.astype(np.float16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8), None


def transform_ndarray_kp_to_cv2_kp(kp_ndarray: np.ndarray) -> List[cv2.KeyPoint]:
    return [cv2.KeyPoint(x=float(point[0]), y=float(point[1]), size=1) for point in kp_ndarray]


def find_descriptors_with_brief(img: MatLike, kp: Tuple[cv2.KeyPoint]):
    brief = cv2.xfeatures2d.BriefDescriptorExtractor().create()
    _kp, descriptors = brief.compute(img, kp)
    return _kp, descriptors


def apply_fast_with_brief(img, **kwargs):
    output_img, kp = apply_fast_detector(img, **kwargs)
    kp, descriptors = find_descriptors_with_brief(img, kp)
    return output_img, kp, descriptors


def apply_harris_with_brief(img, **kwargs):
    output_img, kp_ndarray = apply_harris_detector(img, **kwargs)
    kp = transform_ndarray_kp_to_cv2_kp(kp_ndarray)
    kp, descriptors = find_descriptors_with_brief(img, kp)
    return output_img, kp, descriptors


def calc_evaluators(img, detector_descriptor, augmentation, augmentation_str: str, matcher, rad_th=5.):
    output_img, kp, descriptors = detector_descriptor(img)
    aug_img, additional_data = augmentation(img)
    aug_output_img, aug_kp, aug_descriptors = detector_descriptor(aug_img)

    # cv2.imshow('image', output_img)
    # cv2.waitKey(0)
    # cv2.imshow('image', aug_output_img)
    # cv2.waitKey(0)

    matches = matcher.match(descriptors, aug_descriptors)
    # print(matches)

    matches = sorted(matches, key=lambda x: x.distance)
    kp_matched = np.array([kp[m.queryIdx].pt for m in matches])
    aug_kp_matched = np.array([aug_kp[m.trainIdx].pt for m in matches])

    if 'rotate' in augmentation_str:
        aug_kp_matched = rotate_coordinates_back(aug_kp_matched, additional_data, img.shape[:2])
    if 'scale' in augmentation_str:
        aug_kp_matched = aug_kp_matched / additional_data

    # print(np.hstack((kp_matched, aug_kp_matched)))

    # Calculate Euclidean distance between matched keypoints
    distances = np.linalg.norm(kp_matched - aug_kp_matched, axis=1)

    # Count matches within the threshold
    repeatable_matches = np.sum(distances <= rad_th)
    # print(repeatable_matches)
    repeatability = repeatable_matches / len(kp)

    print(f"Repeatability: {repeatability:.4f}")

    valid_distances = distances[distances <= rad_th]
    if valid_distances.size > 0:
        average_distance = np.mean(valid_distances)
    else:
        average_distance = None
    print("Average Distance", average_distance)

    return {
        "repeatability": repeatability,
        "localization_error": average_distance
    }


augmentations = {
    "gauss_noise_10": lambda img: add_gaussian_noise(img, 0, 10),
    "gauss_noise_20": lambda img: add_gaussian_noise(img, 0, 20),
    "rotate_30": lambda img: rotate_image(img, 30, 1),
    "rotate_70": lambda img: rotate_image(img, 70, 1),
    "scale_half": lambda img: rescale_image(img, 0.5),
    "scale_2": lambda img: rescale_image(img, 2),
    "gauss_filter_5": lambda img: (cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.), None),
    "gauss_filter_9": lambda img: (cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=1.), None),
}

bf_hamm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf_l2norm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

detector_descriptor_matcher = {
    # "harris_brief": (lambda img: apply_harris_with_brief(img), bf_l2norm),
    "orb": (lambda img: apply_orb_detector(img, 200), bf_hamm),
    "fast_brief": (lambda img: apply_fast_with_brief(img), bf_hamm),
    "akaze": (lambda img: apply_akaze_detector(img), bf_l2norm),
}


def run_eval(images_paths):
    #accumulator = defaultdict(dict)
    accumulator = defaultdict(lambda: defaultdict(lambda: {'repeatability': [], 'localization_error': []}))

    for image_path in images_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512, 512))

        for dd_key, (dd, matcher) in detector_descriptor_matcher.items():
            #accumulator[dd_key] = defaultdict(dict)
            print(f"---------\n{dd_key}\n----------")
            for aug_str, aug_func in augmentations.items():
                #accumulator[dd_key][aug_str] = defaultdict(list)
                print(aug_str)
                evaluators = calc_evaluators(img, dd, aug_func, aug_str, matcher)
                accumulator[dd_key][aug_str]['repeatability'].append(evaluators['repeatability'])
                accumulator[dd_key][aug_str]['localization_error'].append(evaluators['localization_error'])

    # with open("accum.pickle", "wb") as f:
    #     pickle.dump(accumulator, f)

    print(accumulator)
    df = accumulator_to_df(accumulator)
    print(df)
    print(df.head())
    print(list(df))

    plt.figure(figsize=(10, 6))

    # Prepare data for plotting repeatability
    repeatability_data = df[
        ['augmentation', 'Algorithm 1_repeatability', 'Algorithm 2_repeatability', 'Algorithm 3_repeatability']]

    # Plot bars for each algorithm
    x = np.arange(len(repeatability_data))  # position for bars on x-axis
    bar_width = 0.2  # Width of bars
    plt.bar(x - bar_width, repeatability_data['Algorithm 1_repeatability'], width=bar_width, label='Algorithm 1',
            color='b')
    plt.bar(x, repeatability_data['Algorithm 2_repeatability'], width=bar_width, label='Algorithm 2', color='g')
    plt.bar(x + bar_width, repeatability_data['Algorithm 3_repeatability'], width=bar_width, label='Algorithm 3',
            color='r')

    # Labeling
    plt.xlabel('Augmentation Type')
    plt.ylabel('Repeatability')
    plt.title('Repeatability for Different Algorithms and Augmentations')
    plt.xticks(x, repeatability_data['augmentation'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting Localization Error (one plot for localization error)
    plt.figure(figsize=(10, 6))

    # Prepare data for plotting localization error
    localization_error_data = df[['augmentation', 'Algorithm 1_localization_error', 'Algorithm 2_localization_error',
                                  'Algorithm 3_localization_error']]

    # Plot bars for each algorithm
    plt.bar(x - bar_width, localization_error_data['Algorithm 1_localization_error'], width=bar_width,
            label='Algorithm 1', color='b')
    plt.bar(x, localization_error_data['Algorithm 2_localization_error'], width=bar_width, label='Algorithm 2',
            color='g')
    plt.bar(x + bar_width, localization_error_data['Algorithm 3_localization_error'], width=bar_width,
            label='Algorithm 3', color='r')

    # Labeling
    plt.xlabel('Augmentation Type')
    plt.ylabel('Localization Error')
    plt.title('Localization Error for Different Algorithms and Augmentations')
    plt.xticks(x, localization_error_data['augmentation'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def accumulator_to_df(accumulator):
    rows = []
    for detector, noise_data in accumulator.items():
        for noise_type, metrics in noise_data.items():
            for metric_name, values in metrics.items():
                for idx, value in enumerate(values):
                    rows.append({
                        'detector': detector,
                        'noise_type': noise_type,
                        'metric': metric_name,
                        'index': idx,
                        'value': value
                    })
    df = pd.DataFrame(rows)
    return df






if __name__ == '__main__':
    images_paths = glob.glob("../data/*")
    run_eval(images_paths)
