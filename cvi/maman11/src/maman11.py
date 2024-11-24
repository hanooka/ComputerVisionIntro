from typing import Tuple

import cv2
import glob

import numpy as np
from cv2.typing import MatLike


def apply_harris_detector(img: MatLike, block_size: int, ksize: int, k: float = 0.04, th_factor=0.02) -> MatLike:
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
    :return: marked/colored image with corners
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # using harris detector implemented by opencv, to detect corners
    corners_resp = cv2.cornerHarris(gray_image, blockSize=block_size, ksize=ksize, k=k)
    # dilate the points (make them more visible)
    corners_resp = cv2.dilate(src=corners_resp, kernel=None)
    threshold = th_factor * corners_resp.max()
    ret_img = img.copy()
    # mark red on copy of original image
    ret_img[corners_resp > threshold] = [0, 0, 255]
    return ret_img


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

    return output_image


if __name__ == '__main__':
    images_paths = glob.glob("../data/*")

    for image_path in images_paths[2:]:
        img = cv2.imread(image_path)
        # img = cv2.resize(img, (512, 512))
        #_img = apply_harris_detector(img, 7, 3, 0.04, 0.02)
        # _img = apply_fast_detector(img, th=20)
        # _img = apply_orb_detector(img, 10)
        # _img = apply_sift_detector(img)
        _img = apply_akaze_detector(img)

        # Display the result
        cv2.imshow('image', _img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
