import os
import gc
import sys
import cv2
import csv

import torch
import random
import asyncio
import numpy as np
import kornia.feature as KF

from tqdm import tqdm
from tqdm.asyncio import tqdm as a_tqdm

from cvi.end_project.src.utils.utils import *


async def compute_error(scene, mkpts0, mkpts1, F, inliers, id1, id2, calib_dict, scaling_dict):
    inlier_kp_1 = mkpts0[inliers]
    inlier_kp_2 = mkpts1[inliers]

    # Compute the essential matrix.
    E, R, T = await asyncio.to_thread(
        ComputeEssentialMatrix, F, calib_dict[id1].K, calib_dict[id2].K, inlier_kp_1, inlier_kp_2)
    q = await asyncio.to_thread(QuaternionFromMatrix, R)
    T = T.flatten()

    # Get the relative rotation and translation between these two cameras, given their R and T in the global reference frame.
    R1_gt, T1_gt = calib_dict[id1].R, calib_dict[id1].T.reshape((3, 1))
    R2_gt, T2_gt = calib_dict[id2].R, calib_dict[id2].T.reshape((3, 1))
    dR_gt = np.dot(R2_gt, R1_gt.T)
    dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
    q_gt = await asyncio.to_thread(QuaternionFromMatrix, dR_gt)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

    # Compute the error for this example.
    err_q, err_t = await asyncio.to_thread(ComputeErrorForOneExample, q_gt, dT_gt, q, T, scaling_dict[scene])
    return err_q, err_t


async def calculate_f_and_error(scene, pair, semaphore, print_lock, calib_dict, images_dict, loftr_scale, scaling_dict,
                                matcher):
    async with semaphore:
        id1, id2 = pair.split('-')

        # We take the minimum between loftr_scale and max(h, w) of current image.
        img_1_rescale = min(loftr_scale, max(images_dict[id1].shape[0], images_dict[id1].shape[1]))
        img_2_rescale = min(loftr_scale, max(images_dict[id2].shape[0], images_dict[id2].shape[1]))

        image_1, img1_h_scale, img1_w_scale = await asyncio.to_thread(
            get_tensor_from_np, images_dict[id1], device, img_1_rescale)
        image_2, img2_h_scale, img2_w_scale = await asyncio.to_thread(
            get_tensor_from_np, images_dict[id2], device, img_2_rescale)

        mkpts0, mkpts1, conf = get_loftr_matches(image_1, image_2, matcher)

        # transforming points to original scale
        mkpts0[:, 0] /= img1_h_scale
        mkpts0[:, 1] /= img1_w_scale
        mkpts1[:, 0] /= img2_h_scale
        mkpts1[:, 1] /= img2_w_scale

        mask = conf > 0.1
        mkpts0 = mkpts0[mask].cpu().numpy()
        mkpts1 = mkpts1[mask].cpu().numpy()

        F, inliers = await asyncio.to_thread(
            cv2.findFundamentalMat, mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.9999, 100000)

        # Compute the error for this example.
        err_q, err_t = await compute_error(scene, mkpts0, mkpts1, F, inliers, id1, id2, calib_dict, scaling_dict)

        async with print_lock:
            print(f'{pair}, err_q={(err_q):.02f} (deg), err_t={(err_t):.02f} (m)', flush=True)

        del mkpts0, mkpts1, image_1, image_2
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "scene": scene,
            "pair": pair,
            "err_q": err_q,
            "err_t": err_t
        }


async def fk_my_life(device, print_lock, semaphore, matcher, loftr_scale, max_pairs_per_scene):
    data_src = '../../data/cv-22928-2025-a-project/train/'
    scaling_dict = {}
    with open(f'{data_src}/scaling_factors.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            scaling_dict[row[1]] = float(row[2])

    # We use two different sets of thresholds over rotation and translation.
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)

    # Save the per-sample errors and the accumulated metric to dictionaries, for later inspection.
    errors = {scene: {} for scene in scaling_dict.keys()}
    mAA = {scene: {} for scene in scaling_dict.keys()}

    async def calculate_mAA():
        for scene in scaling_dict.keys():
            # Load ground truth data.
            calib_dict = LoadCalibration(f'{data_src}/{scene}/calibration.csv')
            # Load all pairs, find those with a co-visibility over 0.1, and subsample them.
            covisibility_dict = ReadCovisibilityData(f'{data_src}/{scene}/pair_covisibility.csv')
            pairs = [pair for pair, covis in covisibility_dict.items() if covis >= 0.5]

            print(
                f'-- Processing scene "{scene}": found {len(pairs)} pairs (will keep {min(len(pairs), max_pairs_per_scene)})',
                flush=True)

            # Subsample the pairs. Note that they are roughly sorted by difficulty (easy ones first),
            # so we shuffle them beforehand: results would be misleading otherwise.
            random.shuffle(pairs)
            pairs = pairs[:max_pairs_per_scene]

            ids = []
            for pair in pairs:
                cur_ids = pair.split('-')
                ids += cur_ids
            ids = list(set(ids))

            images_dict = {}

            for id in tqdm(ids):
                images_dict[id] = cv2.cvtColor(cv2.imread(f'{data_src}/{scene}/images/{id}.jpg'), cv2.COLOR_BGR2RGB)

            max_err_acc_q_new = []
            max_err_acc_t_new = []

            tasks = []
            for counter, pair in enumerate(pairs):
                tasks.append(
                    calculate_f_and_error(scene, pair, semaphore, print_lock, calib_dict, images_dict, loftr_scale,
                                          scaling_dict, matcher))

            results = await asyncio.gather(*tasks)

            for r in results:
                errors[r['scene']][r['pair']] = r['err_q'], r['err_t']

            # Histogram the errors over this scene.
            mAA[scene] = ComputeMaa([v[0] for v in errors[scene].values()], [v[1] for v in errors[scene].values()],
                                    thresholds_q, thresholds_t)

            print(f'\nMean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')

        return mAA

    mAA = await calculate_mAA()
    print('\n------- SUMMARY -------\n')
    for scene in scaling_dict.keys():
        print(f'-- Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
    print(f'\nMean average Accuracy on dataset: {np.mean([mAA[scene][0] for scene in mAA]):.05f}')


if __name__ == '__main__':
    random.seed(1337)

    device = torch.device('cuda')
    matcher = KF.LoFTR()
    matcher = matcher.to(device).eval()

    kwargs = {
        "device": device,
        "matcher": matcher,
        "print_lock": asyncio.Lock(),
        "semaphore": asyncio.Semaphore(10),
        "loftr_scale": 1080,
        "max_pairs_per_scene": 25
    }

    asyncio.run(fk_my_life(**kwargs))

"""
-- Processing scene "sagrada_familia": found 460 pairs (will keep 25)
100%|██████████| 34/34 [00:00<00:00, 150.73it/s]
92031733_3992238201-54163059_5716537618, err_q=0.21 (deg), err_t=0.72 (m)
64223960_5316591967-63135281_5801848783, err_q=0.44 (deg), err_t=0.12 (m)
26362358_4619718880-10875736_10366156803, err_q=0.44 (deg), err_t=0.14 (m)
27632173_1671275972-11122557_224876196, err_q=1.15 (deg), err_t=0.26 (m)
99765221_4300410235-05907640_5576489463, err_q=0.82 (deg), err_t=0.59 (m)
79122962_6371534875-70340840_5145139104, err_q=0.47 (deg), err_t=1.05 (m)
69868392_3604404326-69759293_4547275305, err_q=0.30 (deg), err_t=0.25 (m)
64223960_5316591967-54933845_5249908477, err_q=0.15 (deg), err_t=0.11 (m)
79122962_6371534875-42578175_457568876, err_q=0.79 (deg), err_t=0.13 (m)
27210738_2045590131-10875736_10366156803, err_q=0.13 (deg), err_t=0.17 (m)
79122962_6371534875-77972670_214814210, err_q=3.53 (deg), err_t=3.70 (m)
64223960_5316591967-56240925_168522816, err_q=0.17 (deg), err_t=0.13 (m)
62920559_282422375-06788497_3085353292, err_q=0.19 (deg), err_t=0.17 (m)
97250286_522198503-27210738_2045590131, err_q=0.72 (deg), err_t=0.18 (m)
69868392_3604404326-64212638_4064879957, err_q=0.56 (deg), err_t=0.43 (m)
68153082_7213226096-63135281_5801848783, err_q=0.53 (deg), err_t=0.47 (m)
97370029_2231232779-79122962_6371534875, err_q=0.76 (deg), err_t=0.48 (m)
36009638_559975512-11122557_224876196, err_q=1.09 (deg), err_t=1.18 (m)
21526113_4379776807-12447860_7511693888, err_q=0.67 (deg), err_t=0.33 (m)
78782709_457568230-64223960_5316591967, err_q=0.30 (deg), err_t=0.18 (m)
36009638_559975512-34169301_5802403584, err_q=4.29 (deg), err_t=4.07 (m)
77637372_464905579-12447860_7511693888, err_q=0.38 (deg), err_t=0.27 (m)
97370029_2231232779-64212638_4064879957, err_q=0.58 (deg), err_t=0.25 (m)
62920559_282422375-49331387_5131734970, err_q=0.66 (deg), err_t=1.11 (m)
98653115_457567968-39664712_6371434217, err_q=0.60 (deg), err_t=0.16 (m)

Mean average Accuracy on "sagrada_familia": 0.77600

-- Processing scene "british_museum": found 102 pairs (will keep 25)
100%|██████████| 28/28 [00:00<00:00, 102.75it/s]
93658023_4980549800-73839172_5084383337, err_q=0.55 (deg), err_t=0.24 (m)
93658023_4980549800-02724900_7322693146, err_q=0.62 (deg), err_t=0.07 (m)
77723525_5227836172-71902171_5360301587, err_q=0.76 (deg), err_t=0.47 (m)
03576546_367645446-02724900_7322693146, err_q=0.13 (deg), err_t=0.12 (m)
78242267_4404579754-67471600_1224757581, err_q=0.89 (deg), err_t=0.87 (m)
93658023_4980549800-07041504_41859620, err_q=0.73 (deg), err_t=0.49 (m)
73839172_5084383337-07041504_41859620, err_q=0.40 (deg), err_t=0.08 (m)
93658023_4980549800-59326840_8032269061, err_q=0.85 (deg), err_t=0.59 (m)
51869583_2554605419-14501930_447150094, err_q=2.89 (deg), err_t=2.64 (m)
77723525_5227836172-62553151_839585618, err_q=0.23 (deg), err_t=0.13 (m)
93658023_4980549800-42803655_6429621523, err_q=0.51 (deg), err_t=0.38 (m)
92199010_423632152-75879177_2453112255, err_q=0.82 (deg), err_t=0.27 (m)
93658023_4980549800-77723525_5227836172, err_q=0.17 (deg), err_t=1.58 (m)
90250998_7086081737-51869583_2554605419, err_q=0.27 (deg), err_t=0.23 (m)
75879177_2453112255-73839172_5084383337, err_q=4.23 (deg), err_t=7.34 (m)
75879177_2453112255-57504314_2114264842, err_q=0.75 (deg), err_t=0.68 (m)
69960354_6519615529-59326840_8032269061, err_q=1.23 (deg), err_t=10.22 (m)
92199010_423632152-14501930_447150094, err_q=0.68 (deg), err_t=0.49 (m)
75879177_2453112255-51869583_2554605419, err_q=0.99 (deg), err_t=0.94 (m)
50084748_4116974413-32520806_97760578, err_q=1.40 (deg), err_t=1.17 (m)
92942151_2078102935-01858319_78150445, err_q=1.24 (deg), err_t=1.30 (m)
19254317_158538587-12797227_4382766788, err_q=1.78 (deg), err_t=2.11 (m)
66552192_4348687893-62774028_4348685003, err_q=0.17 (deg), err_t=0.01 (m)
90250998_7086081737-59326840_8032269061, err_q=0.64 (deg), err_t=0.63 (m)
66552192_4348687893-63647221_4349439258, err_q=179.71 (deg), err_t=0.05 (m)

Mean average Accuracy on "british_museum": 0.60800

-- Processing scene "taj_mahal": found 430 pairs (will keep 25)
100%|██████████| 35/35 [00:00<00:00, 180.35it/s]
94491171_8777596299-67076827_2124556406, err_q=0.31 (deg), err_t=1.74 (m)
94491171_8777596299-06957163_3715589743, err_q=179.47 (deg), err_t=0.73 (m)
73814021_2123792469-67076827_2124556406, err_q=179.90 (deg), err_t=0.03 (m)
45682083_3883422633-45203396_7995388605, err_q=1.02 (deg), err_t=10.34 (m)
47991702_2123791171-06957163_3715589743, err_q=0.89 (deg), err_t=0.68 (m)
84913604_5946947259-72404379_2279379196, err_q=0.24 (deg), err_t=0.46 (m)
76120880_3716403566-06957163_3715589743, err_q=0.27 (deg), err_t=1.45 (m)
56103980_5447022142-06957163_3715589743, err_q=0.70 (deg), err_t=2.31 (m)
82182023_330370375-52835843_5017299307, err_q=0.29 (deg), err_t=0.86 (m)
66814359_3715596305-54900410_4709458745, err_q=2.75 (deg), err_t=3.41 (m)
47991702_2123791171-16110390_8195401777, err_q=0.68 (deg), err_t=3.00 (m)
79703922_2154749801-72129360_3815958942, err_q=1.57 (deg), err_t=5.54 (m)
16983842_4447398826-12757988_6488356199, err_q=0.24 (deg), err_t=0.64 (m)
73208406_5735173134-49313326_9719887895, err_q=2.20 (deg), err_t=7.65 (m)
73814021_2123792469-11160569_2123792303, err_q=179.85 (deg), err_t=0.03 (m)
82354295_233206040-54900410_4709458745, err_q=0.39 (deg), err_t=1.28 (m)
81647157_2124556598-64695426_2124556790, err_q=0.06 (deg), err_t=0.09 (m)
97524709_11495119913-63688033_11494540425, err_q=0.22 (deg), err_t=0.40 (m)
87068487_371077410-84635232_4846035710, err_q=0.69 (deg), err_t=2.91 (m)
84913604_5946947259-10149429_5524882485, err_q=0.59 (deg), err_t=0.98 (m)
49313326_9719887895-06957163_3715589743, err_q=0.90 (deg), err_t=1.48 (m)
92061341_4418372309-78300543_3971893732, err_q=0.36 (deg), err_t=2.18 (m)
72404379_2279379196-06957163_3715589743, err_q=0.22 (deg), err_t=1.63 (m)
75663750_3884257134-62851886_320377486, err_q=2.33 (deg), err_t=3.51 (m)
92061341_4418372309-45203396_7995388605, err_q=0.48 (deg), err_t=5.66 (m)

Mean average Accuracy on "taj_mahal": 0.31600

-- Processing scene "lincoln_memorial_statue": found 563 pairs (will keep 25)
100%|██████████| 38/38 [00:00<00:00, 136.67it/s]
92205152_6228122951-43508548_6814413053, err_q=0.15 (deg), err_t=0.05 (m)
78852386_4857290893-63254605_13698608174, err_q=0.08 (deg), err_t=0.02 (m)
74383616_261225765-17318660_7986785718, err_q=0.57 (deg), err_t=0.06 (m)
96661413_593561088-67626870_2229917645, err_q=0.70 (deg), err_t=0.35 (m)
90174531_8978485136-76213469_2907113872, err_q=0.25 (deg), err_t=0.03 (m)
92205152_6228122951-81908799_469363288, err_q=0.46 (deg), err_t=0.04 (m)
97590139_144606968-24619076_4610384086, err_q=0.54 (deg), err_t=0.03 (m)
28493739_7144738253-25063495_13061263145, err_q=0.47 (deg), err_t=0.09 (m)
83774859_4943410776-58783468_6205561152, err_q=1.13 (deg), err_t=0.50 (m)
92205152_6228122951-76213469_2907113872, err_q=0.53 (deg), err_t=0.06 (m)
63458051_2947647269-61021000_13732290265, err_q=7.70 (deg), err_t=2.79 (m)
63254605_13698608174-61021000_13732290265, err_q=0.49 (deg), err_t=0.09 (m)
93851052_9992440633-92205152_6228122951, err_q=0.82 (deg), err_t=0.26 (m)
90174531_8978485136-74715697_1738168028, err_q=1.08 (deg), err_t=0.05 (m)
86336246_6730237565-18350939_3569121838, err_q=0.18 (deg), err_t=0.02 (m)
22081003_5964891946-21335937_8343210636, err_q=0.40 (deg), err_t=0.14 (m)
92840923_5542897481-57816948_8766296745, err_q=1.22 (deg), err_t=0.48 (m)
82564930_2597002115-57816948_8766296745, err_q=0.15 (deg), err_t=0.01 (m)
90556331_7474296214-42025522_2429794709, err_q=0.18 (deg), err_t=0.05 (m)
78409376_330911655-25063495_13061263145, err_q=0.12 (deg), err_t=0.04 (m)
82564930_2597002115-02147562_7180435655, err_q=0.76 (deg), err_t=0.14 (m)
97215532_4943410648-81908799_469363288, err_q=1.21 (deg), err_t=0.45 (m)
82564930_2597002115-06921452_9027346137, err_q=0.25 (deg), err_t=0.06 (m)
94316597_3993626534-25264314_11125027143, err_q=0.73 (deg), err_t=0.02 (m)
85693775_8136575805-43881603_10018525955, err_q=1.42 (deg), err_t=0.27 (m)

Mean average Accuracy on "lincoln_memorial_statue": 0.91200

-- Processing scene "buckingham_palace": found 49 pairs (will keep 25)
100%|██████████| 27/27 [00:00<00:00, 118.08it/s]
40643701_5922718641-32934563_2735018403, err_q=0.44 (deg), err_t=0.19 (m)
68424227_10780155995-40643701_5922718641, err_q=0.31 (deg), err_t=3.36 (m)
68128475_4677074976-31650540_6170454977, err_q=0.13 (deg), err_t=0.18 (m)
59781164_3740564586-05043726_7787106650, err_q=0.12 (deg), err_t=0.22 (m)
91446393_39141513-40643701_5922718641, err_q=3.99 (deg), err_t=6.38 (m)
70391907_3740589694-40643701_5922718641, err_q=1.30 (deg), err_t=3.09 (m)
91136231_4677075766-01973449_5848775157, err_q=1.16 (deg), err_t=2.50 (m)
95489130_3640825724-18742862_7787110970, err_q=0.07 (deg), err_t=0.04 (m)
91136231_4677075766-68128475_4677074976, err_q=179.92 (deg), err_t=0.02 (m)
42572558_3847795752-33061736_676507748, err_q=0.25 (deg), err_t=0.06 (m)
11607713_8957101211-02183697_6387898075, err_q=2.56 (deg), err_t=10.03 (m)
70391907_3740589694-32934563_2735018403, err_q=3.58 (deg), err_t=10.78 (m)
25475150_3097984411-04577578_104950711, err_q=1.26 (deg), err_t=1.94 (m)
92121328_5579434644-29738870_8712579658, err_q=0.66 (deg), err_t=0.02 (m)
91446393_39141513-68424227_10780155995, err_q=0.09 (deg), err_t=0.03 (m)
68424227_10780155995-56366869_8166763434, err_q=0.16 (deg), err_t=0.03 (m)
31650540_6170454977-01973449_5848775157, err_q=0.31 (deg), err_t=0.38 (m)
68424227_10780155995-32934563_2735018403, err_q=0.13 (deg), err_t=0.00 (m)
87785947_1573426398-72810677_4167966956, err_q=1.20 (deg), err_t=2.22 (m)
68424227_10780155995-33061736_676507748, err_q=0.08 (deg), err_t=0.01 (m)
42572558_3847795752-23458590_2731063123, err_q=1.56 (deg), err_t=0.40 (m)
18742862_7787110970-01973449_5848775157, err_q=2.57 (deg), err_t=5.31 (m)
53271175_196359090-13593997_333454483, err_q=0.55 (deg), err_t=1.60 (m)
91446393_39141513-32934563_2735018403, err_q=0.29 (deg), err_t=0.05 (m)
72810677_4167966956-53271175_196359090, err_q=0.95 (deg), err_t=1.68 (m)

Mean average Accuracy on "buckingham_palace": 0.58000

-- Processing scene "trevi_fountain": found 2348 pairs (will keep 25)
100%|██████████| 45/45 [00:00<00:00, 151.09it/s]
73635234_3535400695-49526651_4313316394, err_q=0.23 (deg), err_t=0.20 (m)
90873732_2346257740-76104794_5487278337, err_q=9.87 (deg), err_t=6.32 (m)
75810419_3515709926-24304044_5235207507, err_q=2.92 (deg), err_t=0.32 (m)
43535476_3244521709-28644733_4987378033, err_q=0.35 (deg), err_t=0.16 (m)
35176299_8523343702-04319993_12789831053, err_q=0.71 (deg), err_t=0.86 (m)
95345147_2076154657-41815199_3578791286, err_q=0.65 (deg), err_t=0.64 (m)
75261125_3476174566-09955671_5104777031, err_q=1.10 (deg), err_t=0.48 (m)
56493755_3454995877-49423560_5432154268, err_q=0.81 (deg), err_t=0.40 (m)
76924854_5406455624-74712645_4623084402, err_q=0.16 (deg), err_t=0.93 (m)
18292753_2698612173-13136303_4532536991, err_q=0.29 (deg), err_t=0.07 (m)
49921508_3425495160-24342233_4427781760, err_q=0.95 (deg), err_t=2.14 (m)
81531282_3528611501-45996779_3151318831, err_q=0.45 (deg), err_t=0.04 (m)
96965379_2766035750-81545978_104769022, err_q=2.00 (deg), err_t=0.62 (m)
40891053_7044077771-18959193_201239784, err_q=1.27 (deg), err_t=0.73 (m)
89086898_3494993799-09318341_728794561, err_q=0.56 (deg), err_t=1.01 (m)
78368537_4517640000-72611872_4900919995, err_q=1.50 (deg), err_t=0.84 (m)
72611872_4900919995-47925804_248181354, err_q=0.78 (deg), err_t=1.33 (m)
78010545_349314185-33729878_8818031078, err_q=0.27 (deg), err_t=0.13 (m)
81507347_4644210584-24417070_6262640016, err_q=0.83 (deg), err_t=1.04 (m)
90873732_2346257740-86451018_5200654015, err_q=2.59 (deg), err_t=1.49 (m)
72919927_2696536810-45798320_11687831675, err_q=0.59 (deg), err_t=0.38 (m)
78368537_4517640000-42339817_2799366536, err_q=0.45 (deg), err_t=0.94 (m)
64666251_3271717782-20954938_8692839289, err_q=0.31 (deg), err_t=0.47 (m)
97815951_7600091342-42339817_2799366536, err_q=0.47 (deg), err_t=0.12 (m)
49526651_4313316394-12731064_3885646152, err_q=1.32 (deg), err_t=0.33 (m)

Mean average Accuracy on "trevi_fountain": 0.65600

-- Processing scene "pantheon_exterior": found 932 pairs (will keep 25)
100%|██████████| 36/36 [00:00<00:00, 116.45it/s]
60625441_3395216547-54921153_4622487551, err_q=1.02 (deg), err_t=0.20 (m)
74314530_3698262397-28922342_12853681983, err_q=0.57 (deg), err_t=0.63 (m)
78325458_2795256515-00936980_2039860114, err_q=0.46 (deg), err_t=0.04 (m)
91574264_2032882288-46390865_3455802818, err_q=0.85 (deg), err_t=0.54 (m)
18990907_13666504113-13516917_263156785, err_q=0.54 (deg), err_t=0.61 (m)
98604604_7278562268-36058585_5900767388, err_q=0.16 (deg), err_t=0.09 (m)
69793377_3641938741-50245548_40149591, err_q=0.62 (deg), err_t=5.75 (m)
87537519_5722191686-01971555_12232925533, err_q=1.15 (deg), err_t=0.45 (m)
46390865_3455802818-32426067_492457964, err_q=1.40 (deg), err_t=11.00 (m)
69793377_3641938741-67164249_4622479909, err_q=2.72 (deg), err_t=0.55 (m)
76854619_5283262335-32426067_492457964, err_q=0.50 (deg), err_t=0.20 (m)
80415207_5039424493-62033791_8685765013, err_q=1.12 (deg), err_t=0.04 (m)
87537519_5722191686-38260203_9614860451, err_q=0.60 (deg), err_t=0.09 (m)
76854619_5283262335-13458889_7495894, err_q=0.37 (deg), err_t=0.57 (m)
80405536_2060199187-33256727_3697387465, err_q=0.08 (deg), err_t=0.25 (m)
87962248_5671261635-18990907_13666504113, err_q=1.24 (deg), err_t=0.88 (m)
76854619_5283262335-20475220_8205864020, err_q=0.22 (deg), err_t=0.48 (m)
66507236_3495812128-63115994_3603886857, err_q=0.56 (deg), err_t=0.29 (m)
98604604_7278562268-41798426_9614861035, err_q=0.42 (deg), err_t=0.39 (m)
80200825_9642334654-13516917_263156785, err_q=1.11 (deg), err_t=0.14 (m)
82477194_6031706231-01971555_12232925533, err_q=4.43 (deg), err_t=2.54 (m)
67256000_2696331140-38031597_4190690247, err_q=0.18 (deg), err_t=0.11 (m)
36058585_5900767388-18990907_13666504113, err_q=0.85 (deg), err_t=0.59 (m)
98604604_7278562268-70739776_5998319875, err_q=0.14 (deg), err_t=0.06 (m)
67164249_4622479909-16755971_162292537, err_q=1.13 (deg), err_t=0.14 (m)

Mean average Accuracy on "pantheon_exterior": 0.72000

-- Processing scene "temple_nara_japan": found 18 pairs (will keep 18)
100%|██████████| 24/24 [00:00<00:00, 141.18it/s]
48351530_4726336838-47967162_4510919768, err_q=2.86 (deg), err_t=2.85 (m)
93676080_5065578526-50725988_13983146066, err_q=0.26 (deg), err_t=0.28 (m)
45531204_131225369-31629950_8294669242, err_q=0.38 (deg), err_t=0.83 (m)
65927374_4557268646-62230834_2648290472, err_q=1.38 (deg), err_t=2.53 (m)
69924684_7055038381-54184662_5300113584, err_q=0.18 (deg), err_t=0.09 (m)
38794329_412290216-37195811_3395272994, err_q=0.96 (deg), err_t=0.37 (m)
42794347_3394464069-37195811_3395272994, err_q=179.46 (deg), err_t=0.10 (m)
93676080_5065578526-66238148_3810565587, err_q=1.01 (deg), err_t=1.56 (m)
50725988_13983146066-23609267_9062096127, err_q=0.33 (deg), err_t=0.19 (m)
62480426_6172345615-48351530_4726336838, err_q=1.67 (deg), err_t=1.80 (m)
62480426_6172345615-47967162_4510919768, err_q=3.73 (deg), err_t=1.63 (m)
54731592_3664495285-45531204_131225369, err_q=0.60 (deg), err_t=1.01 (m)
77378214_5191024150-72556787_1880210577, err_q=0.30 (deg), err_t=0.48 (m)
65927374_4557268646-56659519_10877859154, err_q=0.59 (deg), err_t=0.90 (m)
95264086_6694083417-63443304_3580343277, err_q=179.99 (deg), err_t=1.35 (m)
62230834_2648290472-48351530_4726336838, err_q=0.62 (deg), err_t=0.50 (m)
43793474_430380799-03195957_219651147, err_q=4.54 (deg), err_t=2.85 (m)
65927374_4557268646-48351530_4726336838, err_q=0.16 (deg), err_t=0.83 (m)

Mean average Accuracy on "temple_nara_japan": 0.50000

-- Processing scene "notre_dame_front_facade": found 641 pairs (will keep 25)
100%|██████████| 38/38 [00:00<00:00, 151.39it/s]
94320337_218677558-84056637_8972124892, err_q=2.09 (deg), err_t=2.66 (m)
58080350_5194572072-53720490_6006661254, err_q=0.31 (deg), err_t=0.02 (m)
48545849_4506500189-36506915_5079146935, err_q=3.67 (deg), err_t=0.23 (m)
71300674_2388731725-61233613_204929162, err_q=0.42 (deg), err_t=0.06 (m)
97782052_5986763759-88478898_3501423369, err_q=0.85 (deg), err_t=0.07 (m)
99784014_3777253105-36506915_5079146935, err_q=5.04 (deg), err_t=0.26 (m)
40561867_3579019621-36506915_5079146935, err_q=1.16 (deg), err_t=0.19 (m)
73252996_2810972219-33964121_8571233470, err_q=0.27 (deg), err_t=0.18 (m)
63362920_8594646157-20529523_4807251972, err_q=0.14 (deg), err_t=0.07 (m)
78489208_8066294881-16403666_4619513285, err_q=0.69 (deg), err_t=0.05 (m)
97810716_313657629-48545849_4506500189, err_q=4.75 (deg), err_t=0.62 (m)
91656956_4923117684-79222984_9062375931, err_q=0.55 (deg), err_t=0.06 (m)
73823709_2379577727-18481179_2746947984, err_q=0.51 (deg), err_t=0.01 (m)
73929271_3251936784-21013786_4208139268, err_q=0.25 (deg), err_t=0.05 (m)
73929271_3251936784-28398273_5904550418, err_q=1.91 (deg), err_t=0.37 (m)
84056637_8972124892-43380956_8894397834, err_q=0.21 (deg), err_t=0.02 (m)
88494215_13093430663-35763676_5982822151, err_q=0.18 (deg), err_t=0.03 (m)
73929271_3251936784-71300674_2388731725, err_q=0.72 (deg), err_t=0.13 (m)
63362920_8594646157-27273198_5734552964, err_q=0.39 (deg), err_t=0.07 (m)
70950675_2625630826-35763676_5982822151, err_q=1.13 (deg), err_t=0.11 (m)
40561867_3579019621-05903544_2604467668, err_q=0.69 (deg), err_t=0.09 (m)
77617977_133127190-52951096_3427419374, err_q=0.05 (deg), err_t=0.01 (m)
94887785_2665835098-27335203_487553664, err_q=0.37 (deg), err_t=0.01 (m)
90188562_2452517640-05903544_2604467668, err_q=0.81 (deg), err_t=0.23 (m)
05903544_2604467668-03910817_5904551166, err_q=2.50 (deg), err_t=0.56 (m)

Mean average Accuracy on "notre_dame_front_facade": 0.88800

-- Processing scene "colosseum_exterior": found 815 pairs (will keep 25)
100%|██████████| 40/40 [00:00<00:00, 140.34it/s]
58558995_8108191021-10193054_9174141765, err_q=3.55 (deg), err_t=12.90 (m)
95149062_6032288272-26236359_2890221439, err_q=179.94 (deg), err_t=9.83 (m)
95733039_5114469579-92817674_6031728525, err_q=3.89 (deg), err_t=6.39 (m)
73772141_5976578323-13776566_69680703, err_q=1.43 (deg), err_t=9.25 (m)
61865344_5086529333-42928855_2360412631, err_q=1.28 (deg), err_t=2.77 (m)
31608241_8388189859-24699739_5863373729, err_q=3.19 (deg), err_t=13.35 (m)
72483683_5544864265-34966508_3425496532, err_q=1.81 (deg), err_t=6.86 (m)
06891540_2350092518-03841083_3697423443, err_q=1.85 (deg), err_t=6.94 (m)
64968571_2448141464-33729039_5520712742, err_q=0.83 (deg), err_t=1.79 (m)
94818144_2345428455-93398566_312325000, err_q=9.13 (deg), err_t=41.44 (m)
72483683_5544864265-42415433_4102591585, err_q=2.28 (deg), err_t=6.18 (m)
93398566_312325000-27761217_560422976, err_q=1.87 (deg), err_t=21.06 (m)
92723777_5919875791-28989130_4674525415, err_q=1.40 (deg), err_t=12.77 (m)
79719581_6150346823-45414320_5057375643, err_q=0.43 (deg), err_t=0.85 (m)
95149062_6032288272-36840190_2262283891, err_q=47.65 (deg), err_t=88.30 (m)
89244558_6085575451-67392776_6746131399, err_q=1.95 (deg), err_t=9.65 (m)
95733039_5114469579-72483683_5544864265, err_q=0.68 (deg), err_t=0.96 (m)
58918085_739927630-40981306_1504810601, err_q=0.43 (deg), err_t=0.64 (m)
64956914_3388805033-54426674_6691758161, err_q=0.33 (deg), err_t=0.88 (m)
86978670_6994075672-27761217_560422976, err_q=0.68 (deg), err_t=3.14 (m)
89477133_5200565537-69659099_5086527309, err_q=3.50 (deg), err_t=10.57 (m)
92242208_4015307265-21831765_6955993089, err_q=3.66 (deg), err_t=7.30 (m)
96754926_3528571593-94818144_2345428455, err_q=0.50 (deg), err_t=6.97 (m)
92242208_4015307265-58918085_739927630, err_q=1.25 (deg), err_t=2.73 (m)
69091975_5405940535-64956914_3388805033, err_q=0.94 (deg), err_t=1.71 (m)

Mean average Accuracy on "colosseum_exterior": 0.13200

-- Processing scene "sacre_coeur": found 195 pairs (will keep 25)
100%|██████████| 35/35 [00:00<00:00, 87.72it/s]
66666369_72480322-33631460_6836524250, err_q=0.85 (deg), err_t=0.48 (m)
77299481_4479618965-25122990_2171869199, err_q=0.75 (deg), err_t=0.07 (m)
33631460_6836524250-23735654_3093471146, err_q=0.44 (deg), err_t=0.23 (m)
99119091_3503925030-98546957_2200688611, err_q=0.09 (deg), err_t=0.06 (m)
32069695_13023252634-04822190_12621366523, err_q=0.66 (deg), err_t=0.13 (m)
99119091_3503925030-33631460_6836524250, err_q=0.22 (deg), err_t=0.08 (m)
96450192_6766803797-92124082_180660201, err_q=0.09 (deg), err_t=0.07 (m)
76519782_3489095588-39852107_3868092306, err_q=2.17 (deg), err_t=3.00 (m)
93423882_13476569344-07643413_4366433493, err_q=179.95 (deg), err_t=0.09 (m)
96450192_6766803797-66666369_72480322, err_q=0.10 (deg), err_t=0.10 (m)
50715122_352221583-07643413_4366433493, err_q=1.16 (deg), err_t=0.08 (m)
77299481_4479618965-66666369_72480322, err_q=0.49 (deg), err_t=0.30 (m)
97643496_704095933-55597679_2718818848, err_q=0.80 (deg), err_t=0.44 (m)
62960466_112702816-07885041_2904996279, err_q=0.20 (deg), err_t=0.15 (m)
97437832_311711460-68663909_1356782012, err_q=1.82 (deg), err_t=0.60 (m)
82330476_2679804255-26500720_4550353765, err_q=1.70 (deg), err_t=1.83 (m)
32069695_13023252634-10162786_7360656560, err_q=0.20 (deg), err_t=0.23 (m)
90006833_487404833-39172134_3155998144, err_q=0.61 (deg), err_t=0.77 (m)
76645481_5974383848-24390380_2180271939, err_q=0.60 (deg), err_t=0.03 (m)
95734826_713795234-88698102_4801040627, err_q=0.06 (deg), err_t=0.04 (m)
77299481_4479618965-07885041_2904996279, err_q=0.39 (deg), err_t=0.09 (m)
92124082_180660201-77299481_4479618965, err_q=0.39 (deg), err_t=0.05 (m)
70860676_2671019430-59567855_5991079082, err_q=0.14 (deg), err_t=0.09 (m)
90467261_5816977333-39100929_2206793555, err_q=0.44 (deg), err_t=0.18 (m)
66666369_72480322-55597679_2718818848, err_q=0.18 (deg), err_t=0.15 (m)

Mean average Accuracy on "sacre_coeur": 0.82400

-- Processing scene "brandenburg_gate": found 750 pairs (will keep 25)
100%|██████████| 38/38 [00:00<00:00, 100.73it/s]
38600512_2168650655-12644892_6035054052, err_q=4.29 (deg), err_t=2.58 (m)
17262282_1141017004-05712502_4051278060, err_q=1.37 (deg), err_t=0.80 (m)
82795354_2835662202-67266149_8114963267, err_q=0.36 (deg), err_t=0.21 (m)
82795354_2835662202-74616849_13891295722, err_q=1.42 (deg), err_t=2.24 (m)
37403694_5070563509-10624116_2746543073, err_q=0.11 (deg), err_t=0.03 (m)
90135423_286554388-62661350_182339698, err_q=0.86 (deg), err_t=0.98 (m)
73131946_5967189750-65593803_7168341735, err_q=0.48 (deg), err_t=0.35 (m)
78762094_501186144-54233019_1901167706, err_q=0.94 (deg), err_t=0.53 (m)
93505867_8891475944-90142764_8281118641, err_q=1.02 (deg), err_t=0.66 (m)
76144572_5654283700-62661350_182339698, err_q=1.42 (deg), err_t=1.60 (m)
90142764_8281118641-38600512_2168650655, err_q=1.10 (deg), err_t=1.09 (m)
26655336_6192911644-09840944_156579676, err_q=0.73 (deg), err_t=0.37 (m)
82795354_2835662202-27175827_2221395482, err_q=0.19 (deg), err_t=0.10 (m)
90920828_5082887495-76144572_5654283700, err_q=1.47 (deg), err_t=0.80 (m)
91824386_51142061-78762094_501186144, err_q=1.40 (deg), err_t=1.08 (m)
77037144_51142129-10605892_4176363979, err_q=1.53 (deg), err_t=0.97 (m)
95048461_7775852580-49325118_2768773629, err_q=1.16 (deg), err_t=0.99 (m)
37403694_5070563509-31638320_5722213561, err_q=0.34 (deg), err_t=0.24 (m)
79772920_6916144072-27541834_311888773, err_q=1.98 (deg), err_t=1.20 (m)
66720928_3418295149-12644892_6035054052, err_q=2.12 (deg), err_t=1.43 (m)
95850075_1570642878-60770994_853214983, err_q=0.80 (deg), err_t=1.40 (m)
72124445_1515477052-66720928_3418295149, err_q=2.26 (deg), err_t=2.89 (m)
68122503_5083478858-45962310_5550442083, err_q=0.61 (deg), err_t=0.31 (m)
67814984_5083477916-17262282_1141017004, err_q=1.26 (deg), err_t=1.90 (m)
74616849_13891295722-25378482_2036660966, err_q=2.11 (deg), err_t=1.19 (m)

Mean average Accuracy on "brandenburg_gate": 0.57200


------- SUMMARY -------

-- Mean average Accuracy on "sagrada_familia": 0.77600
-- Mean average Accuracy on "british_museum": 0.60800
-- Mean average Accuracy on "taj_mahal": 0.31600
-- Mean average Accuracy on "lincoln_memorial_statue": 0.91200
-- Mean average Accuracy on "buckingham_palace": 0.58000
-- Mean average Accuracy on "trevi_fountain": 0.65600
-- Mean average Accuracy on "pantheon_exterior": 0.72000
-- Mean average Accuracy on "temple_nara_japan": 0.50000
-- Mean average Accuracy on "notre_dame_front_facade": 0.88800
-- Mean average Accuracy on "colosseum_exterior": 0.13200
-- Mean average Accuracy on "sacre_coeur": 0.82400
-- Mean average Accuracy on "brandenburg_gate": 0.57200

Mean average Accuracy on dataset: 0.62367

Process finished with exit code 0

"""