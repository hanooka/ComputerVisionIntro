import asyncio
import os
import cv2
import glob

import numpy as np
import subprocess

import torch
import torchvision
import xgboost as xgb
from cv2 import Mat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import VGG16_Weights
from tqdm import tqdm

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


class SceneryImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        #img = torchvision.io.read_image(img_path)
        #img = (img / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def get_my_vgg16():
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    feature_extractor = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    feature_extractor.eval()
    return feature_extractor

def question2():
    labels = extract_labels(images_paths)
    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.2, shuffle=True, stratify=labels))

    print(f"Processing: {len(train_images_paths)} images..")

    feature_extractor = get_my_vgg16()
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize to ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SceneryImageDataset(image_paths=train_images_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    feature_extractor.to(device)

    all_features = []
    with torch.no_grad():  # No need to compute gradients during inference
        for batch in tqdm(data_loader, desc="Processing batches", unit="batch"):
            batch = batch.to(device)  # Move batch to the device (GPU/CPU)
            features = feature_extractor(batch)  # Get features from the model
            all_features.append(features)

    # Step 10: Combine all features into a single tensor
    all_features = torch.cat(all_features, dim=0)
    print(all_features.shape)  # Expected shape: [num_images, 512]


def is_this_the_real_life():

    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    feature_extractor = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    print(feature_extractor)
    # Set the feature extractor to evaluation mode
    feature_extractor.eval()
    # Image transformation (maintain original size)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize to ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Load an image (original size is maintained)
    image = Image.open(images_paths[0]).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Forward pass through the feature extractor
    with torch.no_grad():
        feat_map = feature_extractor(input_tensor)

    print(feat_map)
    print(feat_map.shape)

def question1():
    labels = extract_labels(images_paths)

    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.2, shuffle=True, stratify=labels))

    loop = asyncio.get_event_loop()

    clf, files_to_features, le = loop.run_until_complete(my_train(train_images_paths, train_labels))
    test_features = loop.run_until_complete(files_to_features(test_images_paths))
    test_labels = le.transform(test_labels)
    dtest = xgb.DMatrix(test_features)
    y_preds = clf.predict(dtest, iteration_range=(0, clf.best_iteration))
    roc_auc = roc_auc_score(test_labels, y_preds, multi_class='ovr', average='macro')
    print(f"MACRO ROC AUC: {roc_auc:.4f}")
    roc_auc = roc_auc_score(test_labels, y_preds, multi_class='ovr', average='micro')
    print(f"MICRO ROC AUC: {roc_auc:.4f}")


def main():
    question2()


if __name__ == '__main__':
    main()
