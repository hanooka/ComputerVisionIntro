from typing import Tuple, Union
import sys

import cv2
import glob

import numpy as np
from cv2.typing import MatLike
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClusterMixin

np.set_printoptions(threshold=sys.maxsize)


from sklearn.cluster import DBSCAN


def rotate_coordinates_with_matrix(points, rotation_matrix):
    """
    Rotate a set of points using a precomputed rotation matrix.

    Parameters:
        points (np.ndarray): Array of points as (x, y).
        rotation_matrix (np.ndarray): 2x3 rotation matrix from cv2.getRotationMatrix2D.

    Returns:
        np.ndarray: Array of rotated points as (x', y').
    """
    # Add a column of ones to the points array for affine transformation
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])

    # Apply the rotation matrix
    rotated_points = np.dot(points_homogeneous, rotation_matrix.T)

    # Return the rotated points
    return np.round(rotated_points).astype(int)

def find_matching_points_kdtree(original_points, augmented_points, tolerance=5.0):
    """
    Finds matches between original and augmented keypoints using a k-d tree.

    Parameters:
        original_points (np.ndarray): Transformed original keypoints.
        augmented_points (np.ndarray): Detected keypoints in the augmented image.
        tolerance (float): Maximum distance to consider a match.

    Returns:
        int: Count of matching points.
    """
    # Build k-d tree from rotated points
    tree = cKDTree(augmented_points)

    # Query for nearest neighbors within the tolerance radius
    match_count = 0
    for point in original_points:
        distances, indices = tree.query(point, k=1, distance_upper_bound=tolerance)
        if distances != np.inf:  # Valid match
            match_count += 1

    return match_count

def rotate_coordinates(points, image_shape, angle):
    """
    Rotate a set of points around a given center.

    Parameters:
        points (np.ndarray): Array of points as (x, y).
        image_shape (tuple): numpy array that will help us calculate center
        angle (float): Rotation angle in degrees.

    Returns:
        np.ndarray: Array of rotated points as integers.
    """
    # Convert angle to radians
    center = image_shape[1]//2, image_shape[0]//2
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # Create rotation matrix
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    shifted_points = points - center
    rotated_points = np.dot(shifted_points, rotation_matrix.T)
    rotated_points += center
    rotated_points = np.clip(np.round(rotated_points), a_min=0, a_max=np.inf).astype(int)

    return rotated_points

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
    print(unique_labels)
    clustered_keypoints = []

    for label in unique_labels:
        if label == -1:  # Noise points, optionally handle separately
            continue
        cluster = keypoints[clustering.labels_ == label]
        centroid = cluster.mean(axis=0)  # Compute the cluster centroid
        clustered_keypoints.append(centroid)

    return np.clip(np.array(clustered_keypoints, dtype=int), a_min=0, a_max=np.inf)

def apply_harris_detector(img: MatLike, block_size: int, ksize: int,
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
) -> MatLike:
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
    key_points = fast.detect(gray_image)
    output_image = cv2.drawKeypoints(img, key_points, None, color=(255, 0, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return output_image


def apply_orb_detector(img: MatLike, max_keypoints: int = 500):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB().create(nfeatures=max_keypoints)
    key_points = orb.detect(gray_image, None)
    output_image = cv2.drawKeypoints(
        img,
        key_points,
        None,
        color=(0, 255, 0),  # Green color for keypoints
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return output_image


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
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    print(keypoints)
    print(len(keypoints))
    keypoints: Tuple[cv2.KeyPoint]
    print(keypoints[0].size)
    print(keypoints[0].octave)
    print(keypoints[0].pt)
    print(keypoints[0].angle)
    print(keypoints[0].response)
    print(keypoints[0].class_id)
    print(type(keypoints[0]))
    print(descriptors[0])
    print(descriptors.shape)

    output_image = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_image, keypoints, descriptors

def rescale_image(img: MatLike, scale) -> MatLike:
    """ Rescales image by scale factor. """
    h, w = img.shape[:2]
    new_w = int(w*scale)
    new_h = int(h*scale)
    return cv2.resize(img, (new_w, new_h))

def rotate_image(img: MatLike, angle: Union[float, int], scale: float=1.0) -> (MatLike, np.ndarray):
    """ Rotating the image, with angle and re-scale factor.
    considering rotated image will create new height/width. """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    orig_r_matrix = rotation_matrix.copy()
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int(h * sin_theta + w * cos_theta)
    new_height = int(h * cos_theta + w * sin_theta)
    rotation_matrix[0, 2] += (new_width - w) / 2
    rotation_matrix[1, 2] += (new_height - w) / 2
    return cv2.warpAffine(img, rotation_matrix, (new_width, new_height)), orig_r_matrix

def add_gaussian_noise(img: MatLike, mean=0, std_dev=2):
    """ Add gaussian noise using normal distribution. """
    noise = np.random.normal(mean, std_dev, img.shape).astype(np.float16)
    noisy_image = img.astype(np.float16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)





if __name__ == '__main__':
    images_paths = glob.glob("../data/ist*")

    for image_path in images_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512, 512))
        #img = add_gaussian_noise(img, 0, 40)
        #img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=1.)

        _img, kp = apply_harris_detector(img, 2, 3, 0.04, 0.02)
        kp = cluster_keypoints(keypoints=kp, eps=5, min_samples=1)
        print(kp)
        print()

        r_img, rotation_matrix = rotate_image(img, 90, 1)
        print(r_img.shape)
        r_img, r_kp = apply_harris_detector(r_img, 2, 3, 0.04, 0.02)

        r_kp = rotate_coordinates_with_matrix(r_kp, rotation_matrix)
        r_kp = cluster_keypoints(r_kp, eps=5, min_samples=1)


        sorted_indices = np.lexsort((r_kp[:, 1], r_kp[:, 0]))
        r_kp = r_kp[sorted_indices]

        print(r_kp)

        match_points = find_matching_points_kdtree(kp, r_kp, 5)
        print(match_points)
        print(len(kp))


        #_img = apply_fast_detector(img, th=20)
        #_img = apply_orb_detector(img, 100)
        #_img = apply_sift_detector(img)
        #_img = apply_akaze_detector(img)

        # Display the result
        cv2.imshow('image', _img)
        cv2.waitKey(0)
        cv2.imshow('image', r_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
