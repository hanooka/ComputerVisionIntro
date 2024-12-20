import asyncio
import os
from typing import Union

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

from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import VGG16_Weights
from tqdm import tqdm

from cvi.maman12.src.nn_architectures import my_vgg, BasicCNN, FastCNN

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


def get_xgb_params(le, additional_params: dict = None):
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
    if additional_params:
        xgb_args.update(additional_params)
    if is_gpu:
        xgb_args.update({"device": "cuda:0"})

    return xgb_args


def extract_labels(images_path):
    return list(map(lambda x: os.path.basename(x).split('_')[0], images_path))


def cluster_descriptors(descriptors: Union[tuple, list], n_clusters=100, sample_pct=1., n_init=5, random_state=None) -> KMeans:
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


async def q1_training(images_paths, labels, n_clusters=100):
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
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            labels = self.labels[idx]
            return img, labels
        else:
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


def q2_training(images_paths, labels, batch_size, feature_extractor, device):
    train_images_paths, val_images_paths, train_labels, val_labels = (
        train_test_split(images_paths, labels, test_size=0.25, shuffle=True, stratify=labels))

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    val_labels = le.transform(val_labels)

    transform = get_transform_obj()

    train_dataset = SceneryImageDataset(image_paths=train_images_paths, transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    feature_extractor.to(device)

    training_features = get_features(train_data_loader, device, feature_extractor)

    val_dataset = SceneryImageDataset(image_paths=val_images_paths, transform=transform)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    val_features = get_features(val_data_loader, device, feature_extractor)

    xgb_args = get_xgb_params(le, {"eta": 0.1})
    dtrain = xgb.DMatrix(training_features, train_labels)
    dval = xgb.DMatrix(val_features, val_labels)
    watchlist = [(dtrain, 'trn'), (dval, 'val')]
    clf = xgb.train(xgb_args, dtrain, num_boost_round=2000, evals=watchlist, early_stopping_rounds=200, verbose_eval=50)
    return clf


def get_features(data_loader: DataLoader, device: torch.device, feature_extractor: torch.nn.Sequential) -> np.ndarray:
    """ Given data loader torch model, doing inference and converting result to np array.
    Data loader already set with data, transformation and batch.
    """
    training_features = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches", unit="batch"):
            batch = batch.to(device)
            features = feature_extractor(batch)
            training_features.append(features)
    training_features = torch.cat(training_features, dim=0)
    training_features = training_features.cpu().numpy()
    return training_features


def question2():
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = extract_labels(images_paths)

    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.2, shuffle=True, stratify=labels))

    feature_extractor = get_my_vgg16()
    clf = q2_training(train_images_paths, train_labels, batch_size, feature_extractor, device)

    transform = get_transform_obj()
    test_dataset = SceneryImageDataset(image_paths=test_images_paths, transform=transform)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    feature_extractor.to(device)

    testing_features = get_features(test_data_loader, device, feature_extractor)

    dtest = xgb.DMatrix(testing_features)
    y_preds = clf.predict(dtest, iteration_range=(0, clf.best_iteration))
    roc_auc = roc_auc_score(test_labels, y_preds, multi_class='ovr', average='macro')
    print(f"MACRO ROC AUC: {roc_auc:.4f}")


def get_transform_obj():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize to ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def question1():
    labels = extract_labels(images_paths)

    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.2, shuffle=True, stratify=labels))

    loop = asyncio.new_event_loop()

    clf, files_to_features, le = loop.run_until_complete(q1_training(train_images_paths, train_labels))
    test_features = loop.run_until_complete(files_to_features(test_images_paths))
    test_labels = le.transform(test_labels)
    dtest = xgb.DMatrix(test_features)
    y_preds = clf.predict(dtest, iteration_range=(0, clf.best_iteration))
    roc_auc = roc_auc_score(test_labels, y_preds, multi_class='ovr', average='macro')
    print(f"MACRO ROC AUC: {roc_auc:.4f}")


def q3_training(images_paths, labels, device, batch_size=32, epochs=5):
    train_images_paths, val_images_paths, train_labels, val_labels = (
        train_test_split(images_paths, labels, test_size=0.25, shuffle=True, stratify=labels))

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    val_labels = le.transform(val_labels)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    loss_func = nn.CrossEntropyLoss()

    model = FastCNN(256, 256, len(le.classes_))#my_vgg(num_classes=len(le.classes_), in_dim=3)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    transform = get_transform_obj()

    train_dataset = SceneryImageDataset(image_paths=train_images_paths, transform=transform, labels=train_labels)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = SceneryImageDataset(image_paths=val_images_paths, transform=transform, labels=val_labels)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    for epoch in range(epochs):
        print(f"Epoch {epoch+1}:")
        model.train()
        running_loss = 0.0
        i = 0
        for _inputs, _labels in train_data_loader:
            i += 1
            _inputs, _labels = _inputs.to(device), _labels.to(device)
            optimizer.zero_grad()
            outputs = model(_inputs)
            loss = loss_func(outputs, _labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Batch {i}/{len(train_data_loader)}, Loss: {running_loss/(i*10):.4f}")

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(train_data_loader):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _inputs, _labels in val_data_loader:
                _inputs, _labels = _inputs.to(device), _labels.to(device)
                outputs = model(_inputs)
                _, predicted = torch.max(outputs, 1)
                total += _labels.size(0)
                correct += (predicted == _labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")



def question3():
    batch_size = 32
    labels = extract_labels(images_paths)

    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.01, shuffle=True, stratify=labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = q3_training(train_images_paths, train_labels, device, batch_size=batch_size, epochs=100)


def main():
    #question2()
    question3()


if __name__ == '__main__':
    main()
