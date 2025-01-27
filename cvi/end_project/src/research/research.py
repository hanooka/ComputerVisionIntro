"""

1) Try run over training data.
2) Print data
3) Try SIFT vs KORNIA
4) Use homography to get the warping matrix so we can rotate the image and find better matches
https://www.youtube.com/watch?v=l_qjO4cM74o
5) Crop

"""

import os
import torch
import kornia.feature as KF

# Input data files are available in the read-only "../input/" directory.
# src = '../data/cv-22928-2025-a-project/train'
#
# val_scenes = []
# for f in os.scandir(src):
#     if f.is_dir():
#         cur_scene = os.path.split(f)[-1]
#         print(f'Found scene "{cur_scene}"" at {f.path}')
#         val_scenes += [cur_scene]
#
#
#
#
# device = torch.device('cuda')
# matcher = KF.LoFTR()
# matcher = matcher.to(device).eval()


import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate example keypoints with 3 clusters
np.random.seed(42)
cluster1 = np.random.randn(100, 2) + [11, 14]
cluster2 = np.random.randn(80, 2) + [15, 10]
cluster3 = np.random.randn(60, 2) + [18, 14]
keypoints = np.vstack((cluster1, cluster2, cluster3))

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=1, min_samples=5)  # Adjust 'eps' and 'min_samples' as needed
labels = dbscan.fit_predict(keypoints)

# Extract unique clusters and their centroids
unique_labels = np.unique(labels[labels != -1])  # Exclude noise (-1)
centroids = []
densities = []

for label in unique_labels:
    cluster_points = keypoints[labels == label]
    centroid = np.mean(cluster_points, axis=0)
    centroids.append(centroid)
    densities.append(len(cluster_points))  # Density = number of points in cluster

# Sort clusters by density (most dense to least dense)
sorted_indices = np.argsort(densities)[::-1]
centroids = np.array(centroids)[sorted_indices]
densities = np.array(densities)[sorted_indices]

# Print results
for i, (centroid, density) in enumerate(zip(centroids, densities)):
    print(f"Cluster {i+1}: Centroid = {centroid}, Density = {density}")

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(keypoints[:, 0], keypoints[:, 1], c=labels, cmap='tab10', s=10, label="Keypoints")
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=100, label="Centroids")
plt.title("Keypoints with Detected Clusters and Centroids")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()