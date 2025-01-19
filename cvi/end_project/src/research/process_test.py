import asyncio
import json
import os
import numpy as np
import cv2
import csv
from glob import glob
import torch
import matplotlib.pyplot as plt
import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF
import gc

from kornia_moons.viz import draw_LAF_matches
from tqdm.asyncio import tqdm

device = torch.device('cuda')
matcher = KF.LoFTR(pretrained='outdoor')
matcher = matcher.to(device).eval()

data_fldr = '../../data/'
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

    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def load_torch_image(fname, device):
    img = cv2.imread(fname)
    scale = 840 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)


async def process(row, semaphore: asyncio.Semaphore):
    async with semaphore:
        sample_id, batch_id, image_1_id, image_2_id = row
        try:
            # Load the images.
            image_1 = await asyncio.to_thread(
                load_torch_image, f'{data_fldr}/test_images/{batch_id}/{image_1_id}.jpg', device)
            image_2 = await asyncio.to_thread(
                load_torch_image, f'{data_fldr}/test_images/{batch_id}/{image_2_id}.jpg', device)
            # print(image_1.shape)
            input_dict = {
                "image0": K.color.rgb_to_grayscale(image_1),
                "image1": K.color.rgb_to_grayscale(image_2)
            }

            with torch.no_grad():
                correspondences = matcher(input_dict)

            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()

            if len(mkpts0) > 7:
                F, inliers = await asyncio.to_thread(
                    cv2.findFundamentalMat, mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.25, 0.99999, 100000)
                inliers = inliers > 0
                try:
                    assert F.shape == (3, 3), 'Malformed F?'
                except AssertionError as e:
                    print(e)
            else:
                F = np.zeros((3, 3))
            return sample_id, F
        except Exception as e:
            print(e)
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
    semaphore = asyncio.Semaphore(35)
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
            gc.collect()
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
