import os
import time
from collections import defaultdict
from typing import Tuple, Union, List, Protocol, Any, Callable, Sequence
import sys

import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cv2.typing import MatLike

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#########################
### HELPERS FUNCTIONS ###
#########################

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


def transform_ndarray_kp_to_cv2_kp(kp_ndarray: np.ndarray) -> List[cv2.KeyPoint]:
    """ Transforming np.ndarray into iterable of cv2 keypoints.
    Originally constracted to handle harris corner detector. """
    return [cv2.KeyPoint(x=float(point[0]), y=float(point[1]), size=1) for point in kp_ndarray]


def find_descriptors_with_brief(img: MatLike, kp: Sequence[cv2.KeyPoint]):
    brief = cv2.xfeatures2d.BriefDescriptorExtractor().create()
    _kp, descriptors = brief.compute(img, kp)
    return _kp, descriptors


def accumulator_to_df(accumulator):
    """ Transforming the accumulator object into a pandas dataframe. """
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
    return pd.DataFrame(rows)


##############################
#### DETECTORS DESCRIPTORS ###
##############################

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
    output_image = cv2.drawKeypoints(img, kp, None,
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


def apply_fast_with_brief(img, **kwargs):
    """ Applying FAST, use FAST kp into BRIEF to get descriptors.
    Not the best approach as we have ORB that use few modification to apply the methods in a better fashion. """
    output_img, kp = apply_fast_detector(img, **kwargs)
    kp, descriptors = find_descriptors_with_brief(img, kp)
    return output_img, kp, descriptors


def apply_harris_with_brief(img, **kwargs):
    """ Was a complete failure. """
    output_img, kp_ndarray = apply_harris_detector(img, **kwargs)
    kp = transform_ndarray_kp_to_cv2_kp(kp_ndarray)
    kp, descriptors = find_descriptors_with_brief(img, kp)
    return output_img, kp, descriptors


#######################################
######## AUGMENTATION FUNCTIONS #######
#######################################


class Augmentation(Protocol):
    is_augmentation: bool  # Ensures the function has the `is_augmentation` attribute

    def __call__(self, img: MatLike, *args, **kwargs) -> (Any, Any):
        ...


# Decorator
def augmentation_function(func):
    func.is_augmentation = True
    return func


@augmentation_function
def rescale_image(img: MatLike, scale) -> (MatLike, None):
    """ Rescales image by scale factor. """
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


@augmentation_function
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


@augmentation_function
def add_gaussian_noise(img: MatLike, mean=0, std_dev=2) -> (MatLike, None):
    """ Add gaussian noise using normal distribution. """
    noise = np.random.normal(mean, std_dev, img.shape).astype(np.float16)
    noisy_image = img.astype(np.float16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8), None


def calc_evaluators(
        img,
        detector_descriptor,
        augmentation: Union[Augmentation, Callable],
        augmentation_str: str,
        matcher: cv2.DescriptorMatcher,
        rad_th: Union[float, int] = 5.) -> dict:
    start_time = time.time()
    output_img, kp, descriptors = detector_descriptor(img)
    aug_img, additional_data = augmentation(img)
    aug_output_img, aug_kp, aug_descriptors = detector_descriptor(aug_img)

    matches = matcher.match(descriptors, aug_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    matcher_precision = calculate_matcher_precision(matches, rad_th)

    calculation_time = time.time() - start_time

    kp_matched = np.array([kp[m.queryIdx].pt for m in matches])
    aug_kp_matched = np.array([aug_kp[m.trainIdx].pt for m in matches])

    # Handle special cases of augmentation with respect to key points.
    if 'rotate' in augmentation_str:
        aug_kp_matched = rotate_coordinates_back(aug_kp_matched, additional_data, img.shape[:2])
    if 'scale' in augmentation_str:
        aug_kp_matched = aug_kp_matched / additional_data

    # Calculate Euclidean distance between matched keypoints
    distances = np.linalg.norm(kp_matched - aug_kp_matched, axis=1)

    # Count matches within the threshold
    repeatable_matches = np.sum(distances <= rad_th)

    repeatability = repeatable_matches / len(kp)

    valid_distances = distances[distances <= rad_th]
    if valid_distances.size > 0:
        average_distance = np.mean(valid_distances)
    else:
        average_distance = None

    print("Average Distance", average_distance)
    print(f"Repeatability: {repeatability:.4f}")
    print(f"Match precision: {matcher_precision:.4f}")
    print(f"Calculation time: {calculation_time:.4f}")

    return {
        "repeatability": repeatability,
        "localization_error": average_distance,
        "match_precision": matcher_precision,
        "calculation_time": calculation_time
    }


def calculate_matcher_precision(matches, rad_th):
    true_positives = 0
    false_positives = 0
    for match in matches:
        if match.distance < rad_th:
            true_positives += 1
        else:
            false_positives += 1
    return true_positives / (true_positives + false_positives)


def run_eval(images_paths, detector_descriptor_matcher, augmentations, image_rescale=None):
    # accumulator = defaultdict(dict)
    accumulator = defaultdict(lambda: defaultdict(lambda: {
        'repeatability': [], 'localization_error': [], 'match_precision': [], 'calculation_time': []}))

    for image_path in images_paths:
        img = cv2.imread(image_path)

        if image_rescale and len(image_rescale) == 2:
            img = cv2.resize(img, image_rescale)

        for dd_key, (dd, matcher) in detector_descriptor_matcher.items():
            print(f"---------\n{dd_key}\n----------")
            for aug_str, aug_func in augmentations.items():
                print(aug_str)
                evaluators = calc_evaluators(img, dd, aug_func, aug_str, matcher)
                for k, v in evaluators.items():
                    accumulator[dd_key][aug_str][k].append(v)

    df = accumulator_to_df(accumulator)
    df.to_csv("accum_df.csv", index=False)

    plot_detector_noise_summary(df, metric_value='repeatability')
    plot_detector_noise_summary(df, metric_value='localization_error')
    plot_detector_noise_summary(df, metric_value='match_precision')
    plot_detector_noise_summary(df, metric_value='calculation_time')


def plot_detector_noise_summary(
        df,
        metric_value='localization_error',
        group_column='detector',
        category_column='noise_type',
        metric_column='metric',
        value_column='value'):
    df = df[df[metric_column] == metric_value]
    df = df.fillna(0)
    # groupby detector and noise_type, calculates the mean for the metric
    summary = df.groupby([group_column, category_column])[value_column].mean().reset_index()
    print(f"{metric_value} summary:\n{summary}")

    # Pivot for plotting: detectors as bars for each noise_type
    pivot_summary = summary.pivot(index=category_column, columns=group_column, values=value_column)

    # Plot the bar chart
    pivot_summary.plot(kind='bar', figsize=(12, 6), width=0.8)

    # Add plot details
    plt.title(f'Average {metric_value.capitalize()} by {group_column.capitalize()} and {category_column.capitalize()}')
    plt.ylabel(f'Average {metric_value.capitalize()}')
    plt.xlabel(category_column.capitalize())
    plt.xticks(rotation=45)
    plt.legend(title=group_column.capitalize())
    plt.tight_layout()
    plt.show()


def main(**args):
    """ augmentations and detector/descriptors are defined here, hardcoded. One can add/remove
    as long as you keep the current structure. """
    augmentations = {
        "gauss_noise_10": lambda img: add_gaussian_noise(img, 0, 15),
        "gauss_noise_20": lambda img: add_gaussian_noise(img, 0, 40),
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
    glob_path = os.path.abspath(os.path.join(__file__, "../../data", '*'))
    images_paths = glob.glob(glob_path)
    run_eval(images_paths, detector_descriptor_matcher, augmentations, **args)


if __name__ == '__main__':
    """ 
    You can set args or remove it completely if you want images to be untouched.
    """
    args = {
        "image_rescale": (512, 512)
    }
    main(**args)
