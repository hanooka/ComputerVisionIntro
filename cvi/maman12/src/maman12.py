import asyncio
import os
import cv2
import glob

import numpy as np
import subprocess
import xgboost as xgb
from cv2 import Mat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

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


def get_xgb_params(le):
    is_gpu = subprocess.run(
        ['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).returncode == 0
    xgb_args = {
        "booster": "gbtree",
        "eta": 0.08,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1,
        "alpha": 0.2,
        "tree_method": "hist",
        "objective": "multi:softprob",
        "num_class": len(le.classes_),
        "eval_metric": ["auc", "mlogloss"]
    }
    if is_gpu:
        xgb_args.update({"device": "cuda:0"})

    return xgb_args


def extract_labels(images_path):
    return list(map(lambda x: os.path.basename(x).split('_')[0], images_path))


def cluster_descriptors(descriptors: tuple, n_clusters=100, sample_pct=1., n_init=5, random_state=None) -> KMeans:
    stacked_descriptors = np.vstack(descriptors)
    num_to_sample = int(sample_pct * stacked_descriptors.shape[0])
    sampled_indices = np.random.choice(stacked_descriptors.shape[0], num_to_sample, replace=False)
    sampled_descriptors = stacked_descriptors[sampled_indices]

    print(f"Building {n_clusters} clusters using {sampled_descriptors.shape} descriptors...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans.fit(sampled_descriptors)
    print("Clusters complete")
    return kmeans


async def extract_sift_descriptors(semaphore: asyncio.Semaphore, sift: cv2.Feature2D, img_path: os.path):
    async with semaphore:
        img = await asyncio.to_thread(cv2.imread, img_path)
        _, descriptor = await asyncio.to_thread(sift.detectAndCompute, img, None)
        return descriptor


async def get_data_features(paths: list, sift: cv2.Feature2D, semaphore: asyncio.Semaphore, kmeans: KMeans):
    tasks = [extract_sift_descriptors(semaphore, sift, img_path) for img_path in paths]
    print("Extracting SIFTs...")
    _descriptors = await asyncio.gather(*tasks)
    print("Finished extracting SIFTs.")
    features = []

    for img_descriptors in _descriptors:
        image_clusters = kmeans.predict(img_descriptors)
        histogram, _ = np.histogram(image_clusters, bins=np.arange(kmeans.n_clusters + 1))
        normalized_histogram = normalize(histogram.reshape(1, -1), norm='l1')[0]
        features.append(normalized_histogram)
    return features


async def my_train(images_paths, labels, n_clusters=100):
    """
    bla bla bla
    :param images_paths:
    :param labels:
    :param n_clusters:
    :return: returns a classifier, transform_function, label_encoder
    """
    train_images_paths, val_images_paths, train_labels, val_labels = (
        train_test_split(images_paths, labels, test_size=0.25, shuffle=True, stratify=labels))

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)

    sift = cv2.SIFT().create()
    semaphore = asyncio.Semaphore(100)
    tasks = [extract_sift_descriptors(semaphore, sift, img_path) for img_path in train_images_paths]

    print("Extracting SIFTs...")
    train_descriptors = await asyncio.gather(*tasks)
    print("Finished extracting SIFTs.")

    kmeans = cluster_descriptors(train_descriptors, n_clusters=n_clusters, sample_pct=0.2, n_init=4)

    training_features = []
    for img_descriptors in train_descriptors:
        image_clusters = kmeans.predict(img_descriptors)
        histogram, _ = np.histogram(image_clusters, bins=np.arange(n_clusters + 1))
        normalized_histogram = normalize(histogram.reshape(1, -1), norm='l1')[0]
        training_features.append(normalized_histogram)

    val_features = await get_data_features(val_images_paths, sift, semaphore, kmeans)
    files_to_features = lambda x: get_data_features(x, sift, semaphore, kmeans)
    val_labels = le.transform(val_labels)

    xgb_args = get_xgb_params(le)
    dtrain = xgb.DMatrix(training_features, train_labels)
    dval = xgb.DMatrix(val_features, val_labels)
    watchlist = [(dtrain, 'trn'), (dval, 'val')]
    clf = xgb.train(xgb_args, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=200, verbose_eval=50)
    return clf, files_to_features, le


def main():
    labels = extract_labels(images_paths)

    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.2, shuffle=True, stratify=labels))

    loop = asyncio.get_event_loop()

    clf, files_to_features, le = loop.run_until_complete(my_train(train_images_paths, train_labels))
    test_features = loop.run_until_complete(files_to_features(test_images_paths))
    test_labels = le.transform(test_labels)
    dtest = xgb.DMatrix(test_features, test_labels)
    y_preds = clf.predict(dtest, iteration_range=(0, clf.best_iteration))
    print(y_preds)
    roc_auc = roc_auc_score(test_labels, y_preds, multi_class='ovr', average='macro')
    print(f"MACRO ROC AUC: {roc_auc:.4f}")
    roc_auc = roc_auc_score(test_labels, y_preds, multi_class='ovr', average='micro')
    print(f"MICRO ROC AUC: {roc_auc:.4f}")


if __name__ == '__main__':
    main()
