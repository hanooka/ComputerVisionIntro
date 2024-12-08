import asyncio
import os
import cv2
import glob

import numpy as np
from cv2 import Mat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

def extract_labels(images_path):
    return list(map(lambda x: os.path.basename(x).split('_')[0], images_path))

async def extract_sift_features(sift: cv2.SIFT, img_path: os.path):
    img = await asyncio.to_thread(cv2.imread, img_path)
    _, descriptor = await asyncio.to_thread(sift.detectAndCompute, img, None)
    return descriptor

def my_train(images_paths, labels):
    train_images_paths, train_labels, val_images_paths, val_labels = (
        train_test_split(images_paths, labels, test_size=0.25, shuffle=True, stratify=labels))
    sift = cv2.SIFT().create()
    features = []
    for img_path in train_images_paths:
        desc = extract_sift_features(sift, img_path)
        features.append(desc)
    features = np.array(features)
    print(features)
    print(features.shape)



def main():
    labels = extract_labels(images_paths)
    #le = LabelEncoder()
    #labels = le.fit_transform(labels)

    train_images_paths, test_images_paths, train_labels, test_labels = (
        train_test_split(images_paths, labels, test_size=0.2, shuffle=True, stratify=labels))

    my_train(train_images_paths, train_labels)

    print(len(train_images_paths))
    print(len(test_images_paths))
    print(len(train_labels))
    print(len(test_labels))

if __name__ == '__main__':
    main()

