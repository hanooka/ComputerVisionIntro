#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# based  on - https://www.kaggle.com/code/eduardtrulls/imc2022-baseline-submission-sift

# In[5]:


import os
import numpy as np
import cv2
import csv
from glob import glob
import matplotlib.pyplot as plt


# In[6]:


# If enabled, the notebook will return some feedback and draw images. 
# Set to False before submitting.
dry_run = False


# In[7]:


# Definitions.

def ExtractSiftFeatures(image, detector, num_features):
    '''Compute SIFT features for a given image.'''
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Note that you may actually get more than num_features features, as a feature for one point can have multiple orientations (this is rare).    
    return detector.detectAndCompute(gray, None)[:num_features]


def ArrayFromCvKps(kps):
    '''Convenience function to convert OpenCV keypoints into a simple numpy array.'''
    
    return np.array([kp.pt for kp in kps])


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def BuildCompositeImage(im1, im2, axis=1, margin=0, background=1):
    '''Convenience function to stack two images with different sizes.'''
    
    if background != 0 and background != 1:
        background = 1
    if axis != 0 and axis != 1:
        raise RuntimeError('Axis must be 0 (vertical) or 1 (horizontal')

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3), dtype=np.uint8) + 255 * background
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3), dtype=np.uint8) + 255 * background
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin
    composite[voff1:voff1 + h1, hoff1:hoff1 + w1, :] = im1
    composite[voff2:voff2 + h2, hoff2:hoff2 + w2, :] = im2

    return (composite, (voff1, voff2), (hoff1, hoff2))


def DrawMatches(im1, im2, kp1, kp2, matches, axis=1, margin=0, background=0, linewidth=2):
    '''Draw keypoints and matches.'''
    
    composite, v_offset, h_offset = BuildCompositeImage(im1, im2, axis, margin, background)

    # Draw all keypoints.
    for coord_a, coord_b in zip(kp1, kp2):
        composite = cv2.drawMarker(composite, (int(coord_a[0] + h_offset[0]), int(coord_a[1] + v_offset[0])), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
        composite = cv2.drawMarker(composite, (int(coord_b[0] + h_offset[1]), int(coord_b[1] + v_offset[1])), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
    
    # Draw matches, and highlight keypoints used in matches.
    for idx_a, idx_b in matches:
        composite = cv2.drawMarker(composite, (int(kp1[idx_a, 0] + h_offset[0]), int(kp1[idx_a, 1] + v_offset[0])), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        composite = cv2.drawMarker(composite, (int(kp2[idx_b, 0] + h_offset[1]), int(kp2[idx_b, 1] + v_offset[1])), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        composite = cv2.line(composite,
                             tuple([int(kp1[idx_a][0] + h_offset[0]),
                                   int(kp1[idx_a][1] + v_offset[0])]),
                             tuple([int(kp2[idx_b][0] + h_offset[1]),
     

                              int(kp2[idx_b][1] + v_offset[1])]), color=(0, 0, 255), thickness=1)
    return composite


# In[8]:


# Read the pairs file.

src = '/kaggle/input/cv-22928-2025-a-project'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


# In[9]:


num_features = 8000

# SIFT feature detector.
# We lower the detection threshold to extract a "fixed" number of features -- small images may not be able to reach the budget otherwise.
detector = cv2.SIFT_create(num_features, contrastThreshold=-10000, edgeThreshold=-10000)

# Brute-force matcher with bi-directionaly check.
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Compute this many samples, and fill the rest with random values, to generate a quick submission and check it works without waiting for a full run. Set to -1 to use all samples.
# how_many_to_fill = 500
how_many_to_fill = -1

F_dict = {}
for i, row in enumerate(test_samples):

    if i % 1000 == 0:
        print(i)
        
    sample_id, batch_id, image_1_id, image_2_id = row
    
    if how_many_to_fill >= 0 and i >= how_many_to_fill:
        F_dict[sample_id] = np.random.rand(3, 3)
        continue
    
    # Load the images.
    image_1 = cv2.cvtColor(cv2.imread(f'{src}/test_images/{batch_id}/{image_1_id}.jpg'), cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(cv2.imread(f'{src}/test_images/{batch_id}/{image_2_id}.jpg'), cv2.COLOR_BGR2RGB)
    
    # Extract features.
    keypoints_1, descriptors_1 = ExtractSiftFeatures(image_1, detector, num_features)
    keypoints_2, descriptors_2 = ExtractSiftFeatures(image_2, detector, num_features)
    
    # Compute matches.
    cv_matches = bf.match(descriptors_1, descriptors_2)
    
    # Compute fundamental matrix.
    cur_kp_1 = ArrayFromCvKps(keypoints_1)
    cur_kp_2 = ArrayFromCvKps(keypoints_2)
    matches = np.array([[m.queryIdx, m.trainIdx] for m in cv_matches])
    F, inlier_mask = cv2.findFundamentalMat(cur_kp_1[matches[:, 0]], cur_kp_2[matches[:, 1]], cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, confidence=0.99999, maxIters=100000)
    F_dict[sample_id] = F
    
    if dry_run:
        matches_after_ransac = np.array([match for match, is_inlier in zip(matches, inlier_mask) if is_inlier])
        im_inliers = DrawMatches(image_1, image_2, cur_kp_1, cur_kp_2, matches_after_ransac)
        fig = plt.figure(figsize=(15, 15))
        plt.title(f'{image_1_id}-{image_2_id}')
        plt.imshow(im_inliers)
        plt.axis('off')
        plt.show()

with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')

# !head submission.csv


# In[ ]:




