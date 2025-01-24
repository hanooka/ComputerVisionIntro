import asyncio
import json
import os
import numpy as np
import cv2
import csv
from glob import glob
import torch
import torchvision
import matplotlib.pyplot as plt
import kornia
from cv2 import Mat
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF
import gc

from kornia_moons.viz import draw_LAF_matches
from tqdm.asyncio import tqdm

device = torch.device('cuda')
matcher = KF.LoFTR(pretrained='outdoor')
matcher = matcher.to(device).eval()

data_fldr = '../../data/cv-22928-2025-a-project/'
output_file = 'submission.csv'
checkpoint_file = 'submission_checkpoint.json'

test_samples = []

with open(f'{data_fldr}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''

    if isinstance(M, np.ndarray):
        return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])
    else:
        return M


def get_loftr_matches(img1, img2, matcher):
    """ Given img1, img2 and matcher (which is LoFTR torch.model)
    returns the matching key points 0, 1, and confidence by LoFTR. """
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }
    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0']
    mkpts1 = correspondences['keypoints1']
    conf = correspondences['confidence']

    return mkpts0, mkpts1, conf


def get_tensor_from_np(img: Mat, device, re_scale=1080) -> (torch.Tensor, float, float):
    """ given image and rescale, calculates scale factor. resizing and converting to tensor.
    returns the tensor together with the scales to transpose back to original scale. """
    scale = re_scale / max(img.shape[0], img.shape[1])
    h = int(img.shape[0] * scale)
    w = int(img.shape[1] * scale)
    scale_h = h / img.shape[0]
    scale_w = w / img.shape[1]
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, keepdim=False).float() / 255
    # cv2 imread, converted image to BGR. here we transform to RGB.
    img = K.color.bgr_to_rgb(img)
    return img.to(device), scale_h, scale_w


async def process(row, semaphore: asyncio.Semaphore):
    async with semaphore:
        sample_id, batch_id, image_1_id, image_2_id = row
        try:
            # Load the images.
            img_1 = await asyncio.to_thread(cv2.imread, f'{data_fldr}/test_images/{batch_id}/{image_1_id}.jpg')
            img_2 = await asyncio.to_thread(cv2.imread, f'{data_fldr}/test_images/{batch_id}/{image_2_id}.jpg')
            image_1, img1_h_scale, img1_w_scale = get_tensor_from_np(img_1, device, 1080)
            image_2, img2_h_scale, img2_w_scale = get_tensor_from_np(img_2, device, 1080)

            mkpts0, mkpts1, conf = get_loftr_matches(image_1, image_2, matcher)

            # transforming points to original scale
            mkpts0[:, 0] /= img1_h_scale
            mkpts0[:, 1] /= img1_w_scale
            mkpts1[:, 0] /= img2_h_scale
            mkpts1[:, 1] /= img2_w_scale

            mask = conf > 0.1
            mkpts1 = mkpts1[mask].cpu().numpy()
            mkpts0 = mkpts0[mask].cpu().numpy()

            if len(mkpts0) >= 8:
                F, inliers = await asyncio.to_thread(
                    cv2.findFundamentalMat, mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.99999, 100000)
                #inliers = inliers > 0
                try:
                    assert F.shape == (3, 3), 'Malformed F?'
                except AssertionError as e:
                    print(e)
            else:
                F = np.zeros((3, 3))

            del image_1, image_2
            torch.cuda.empty_cache()
            gc.collect()
            return sample_id, F

        except Exception as e:
            print(e)
            del image_1, image_2
            torch.cuda.empty_cache()
            gc.collect()
            return sample_id, np.zeros((3, 3))


def save_checkpoint(batch_index, processed_samples):
    """Save the current batch index and results to a checkpoint."""
    # Use FlattenMatrix to serialize numpy arrays
    processed_samples_serializable = [
        (sample_id, FlattenMatrix(F)) for sample_id, F in processed_samples
    ]
    checkpoint_data = {
        "batch_index": batch_index,
        "processed_samples": processed_samples_serializable
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)


def load_checkpoint():
    """Load the checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"batch_index": 0, "processed_samples": []}


async def main():
    semaphore = asyncio.Semaphore(10)
    batch_size = 1000

    # Load checkpoint
    checkpoint = load_checkpoint()
    start_batch = checkpoint["batch_index"]
    processed_samples = checkpoint["processed_samples"]

    # Resume from the last processed batch
    if processed_samples:
        print(f"Resuming from batch {start_batch + 1}...")

    with open(output_file, "a" if processed_samples else "w") as f:
        if not processed_samples:
            f.write('sample_id,fundamental_matrix\n')

        for batch_index in range(start_batch, len(test_samples) // batch_size + 1):
            start_idx = batch_index * batch_size
            end_idx = min((batch_index + 1) * batch_size, len(test_samples))
            batch_samples = test_samples[start_idx:end_idx]

            # Process the batch
            tasks = [process(row, semaphore) for row in batch_samples]
            results = await tqdm.gather(*tasks)

            # Write results to the output file
            for sample_id, F in results:
                f.write(f'{sample_id},{FlattenMatrix(F)}\n')

            # Update checkpoint
            save_checkpoint(batch_index + 1, processed_samples + results)

            print(f"Completed batch {batch_index + 1}.")


# async def main():
#     semaphore = asyncio.Semaphore(35)
#     tasks = []
#     for i, row in enumerate(test_samples):
#         tasks.append(process(row, semaphore))
#
#     results = await tqdm.gather(*tasks)
#
#     with open('submission.csv', 'w') as f:
#         f.write('sample_id,fundamental_matrix\n')
#         for sample_id, F in results:
#             f.write(f'{sample_id},{FlattenMatrix(F)}\n')


# Refactor to process in batches.

if __name__ == '__main__':
    asyncio.run(main())
