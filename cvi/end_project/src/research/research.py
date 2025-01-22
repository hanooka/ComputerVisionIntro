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
src = '../data/cv-22928-2025-a-project/train'

val_scenes = []
for f in os.scandir(src):
    if f.is_dir():
        cur_scene = os.path.split(f)[-1]
        print(f'Found scene "{cur_scene}"" at {f.path}')
        val_scenes += [cur_scene]




device = torch.device('cuda')
matcher = KF.LoFTR()
matcher = matcher.to(device).eval()
