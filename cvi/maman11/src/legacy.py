"""
Legacy code that was developed during the process.
No longer used/needed. """


def find_harris_detectors_with_rotation():
    _img, kp = apply_harris_detector(img, 2, 3, 0.04, 0.02)
    kp = cluster_keypoints(keypoints=kp, eps=5, min_samples=1)

    r_img, rotation_matrix = rotate_image(img, 90, 1)
    r_img, r_kp = apply_harris_detector(r_img, 2, 3, 0.04, 0.02)

    r_kp = rotate_coordinates_back(r_kp, rotation_matrix, img.shape[:2])
    r_kp = cluster_keypoints(r_kp, eps=5, min_samples=1)

    # Sorting them so it'll be easier to compare by "eye"
    sorted_indices = np.lexsort((r_kp[:, 1], r_kp[:, 0]))
    r_kp = r_kp[sorted_indices]

    match_points = find_matching_points_kdtree(kp, r_kp, 5)
    # print(match_points)
    # print(len(kp))


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
    center = image_shape[1] // 2, image_shape[0] // 2
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


def compute_repeatability_and_error(orig_kps, rotated_kps, max_distance=10):
    """
    Compute keypoint repeatability and localization error

    Args:
    - orig_kps: Original keypoints
    - rotated_kps: Transformed rotated keypoints
    - max_distance: Maximum allowed distance for matching

    Returns:
    - Repeatability and average localization error
    """
    # Compute pairwise distances
    distances = np.zeros((len(orig_kps), len(rotated_kps)))
    for i, orig_kp in enumerate(orig_kps):
        for j, rot_kp in enumerate(rotated_kps):
            distances[i, j] = np.sqrt(
                (orig_kp.pt[0] - rot_kp.pt[0]) ** 2 +
                (orig_kp.pt[1] - rot_kp.pt[1]) ** 2
            )

    # Find matches within max_distance
    min_distances = distances.min(axis=1)
    matched_mask = min_distances < max_distance

    # Compute metrics
    repeatability = matched_mask.mean()
    avg_localization_error = min_distances[matched_mask].mean()

    return repeatability, avg_localization_error


def main():
    for image_path in images_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512, 512))

        # img = add_gaussian_noise(img, 0, 40)
        # img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=1.)

        _img, kp = apply_fast_detector(img=img, th=20)
        _orig_kp, orig_descriptors = find_descriptors_with_brief(img, kp)

        rotated_img, rotation_matrix = rotate_image(img, 30, 1)

        rot_img, rot_kp = apply_fast_detector(img=rotated_img, th=20)
        _rot_kp, rot_descriptors = find_descriptors_with_brief(img, rot_kp)

        # Transform rotated keypoints back to original coordinates
        transformed_kps = transform_keypoints(rot_kp, rotation_matrix, img.shape)

        # Compute repeatability and localization error
        repeatability, localization_error = compute_repeatability_and_error(
            _orig_kp, transformed_kps
        )

        print(f"Repeatability: {repeatability}")
        print(f"Average Localization Error: {localization_error}")

        print(len(_rot_kp))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(orig_descriptors, rot_descriptors)

        matches = sorted(matches, key=lambda x: x.distance)

        keypoints1_matched = np.array([_orig_kp[m.queryIdx].pt for m in matches])
        keypoints2_matched = np.array([rot_kp[m.trainIdx].pt for m in matches])

        keypoints2_matched = rotate_coordinates_back(keypoints2_matched, rotation_matrix, img.shape[:2])

        print(keypoints1_matched)
        print(keypoints2_matched)

        rad_th = 5.

        # Calculate Euclidean distance between matched keypoints
        distances = np.linalg.norm(keypoints1_matched - keypoints2_matched, axis=1)

        # Count matches within the threshold
        repeatable_matches = np.sum(distances <= rad_th)
        print(repeatable_matches)
        repeatability = repeatable_matches / len(_orig_kp)
        print(f"Repeatability: {repeatability:.4f}")

        # _img = apply_orb_detector(img, 100)
        # _img = apply_sift_detector(img)
        # _img, _, _ = apply_akaze_detector(img)

        # Display the result
        cv2.imshow('image', _img)
        cv2.waitKey(0)
        cv2.imshow('image', rot_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
