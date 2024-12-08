import asyncio
import os
import cv2
import glob

import numpy as np
from cv2 import Mat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

images_path = os.path.join('../data/images')
images_paths = glob.glob(os.path.join(images_path, '*.jpg'))


def example_of_sift():
    img_path = os.path.join(images_path, "mountain_nat783.jpg")
    img = cv2.imread(img_path)

    sift = cv2.SIFT().create()
    kp, descs = sift.detectAndCompute(img, None)

    print(len(kp))
    print(descs.shape)

    # cv2.imshow("lol", img)
    # cv2.waitKey(0)
    img_with_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("lol", img_with_kp)
    cv2.waitKey(0)


def get_pca(descriptors: list):
    descriptors = np.vstack(descriptors)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(descriptors)
    n_components = 50
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(normalized_features)
    explained_variance = pca.explained_variance_ratio_
    print(f"Total explained variance by {n_components} components: {sum(explained_variance):.2f}")
    return pca_features


def extract_labels(images_path):
    return list(map(lambda x: os.path.basename(x).split('_')[0], images_path))


def cluster_descriptors(descriptors: tuple, n_clusters=100, random_state=None):
    stacked_descriptors = np.vstack(descriptors)
    num_to_sample = int(0.2 * stacked_descriptors.shape[0])
    sampled_indices = np.random.choice(stacked_descriptors.shape[0], num_to_sample, replace=False)
    sampled_descriptors = stacked_descriptors[sampled_indices]

    print(f"Building {n_clusters} clusters using {sampled_descriptors.shape} descriptors...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    kmeans.fit(sampled_descriptors)
    print("Clusters complete")
    return kmeans


async def extract_sift_features(semaphore: asyncio.Semaphore, sift: cv2.SIFT, img_path: os.path):
    async with semaphore:
        img = await asyncio.to_thread(cv2.imread, img_path)
        _, descriptor = await asyncio.to_thread(sift.detectAndCompute, img, None)
        return descriptor


async def my_train(images_paths, labels):
    train_images_paths, train_labels, val_images_paths, val_labels = (
        train_test_split(images_paths, labels, test_size=0.25, shuffle=True, stratify=labels))

    sift = cv2.SIFT().create()
    semaphore = asyncio.Semaphore(100)
    tasks = [extract_sift_features(semaphore, sift, img_path) for img_path in train_images_paths]

    print("Extracting SIFTs...")
    descriptors = await asyncio.gather(*tasks)
    print("Finished extracting SIFTs.")

    kmeans = cluster_descriptors(descriptors)
    print(kmeans)


def main():
    labels = extract_labels(images_paths)
    # le = LabelEncoder()
    # labels = le.fit_transform(labels)

    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.2, shuffle=True, stratify=labels))

    asyncio.run(my_train(train_images_paths, train_labels))


if __name__ == '__main__':
    main()
