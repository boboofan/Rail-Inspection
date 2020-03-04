import os
import uuid
import random
import shutil

import numpy as np
import pandas as pd
import cv2

VALID_SPLIT = 0.2
CLASSES = [1, 2, 3, 4, 10, 11, 12, 13]
SCALE = 10

data_root = os.path.join('data', 'images4cls')
shutil.rmtree('data/images4cls/', ignore_errors=True)

os.makedirs(os.path.join(data_root, 'train'))
os.makedirs(os.path.join(data_root, 'valid'))
for cls in CLASSES:
    os.makedirs(os.path.join(data_root, str(cls)))

data_files = os.listdir('data/')
data_files = set([file[:-4] for file in data_files if file.endswith('.png')])

for name in data_files:
    print(f'[INFO] processing {name}')

    try:
        if not os.path.exists(os.path.join('data', name + '.png')) or not os.path.exists(os.path.join('data', name + '.txt')):
            continue

        img = cv2.imread(os.path.join('data', name + '.png'))
        h, w = img.shape[:2]
        ground_truth = pd.read_csv(os.path.join('data', name + '.txt'), header=None).values
        if len(ground_truth) == 0:
            continue
    except:
        continue

    for gt in ground_truth:
        min_x, min_y, max_x, max_y, label = gt
        if min_x >= max_x or min_y >= max_y:
            continue
        array = img[h - max_y:h - min_y + 1, min_x:max_x + 1, :]
        np.save(os.path.join(data_root, str(label), uuid.uuid4().hex), array)

# split the dataset into train set and valid set
print('[INFO] splitting train and valid sets')
for cls in CLASSES:

    cls_path = os.path.join(data_root, str(cls))

    samples = os.listdir(cls_path)
    samples = [os.path.join(cls_path, sample) for sample in samples]

    nb_valid = int(VALID_SPLIT * len(samples))
    nb_train = len(samples) - nb_valid
    random.shuffle(samples)

    target_dir = os.path.join(data_root, 'train', str(cls))
    os.makedirs(target_dir)
    for sample in samples[:nb_train]:
        shutil.move(sample, target_dir)

    target_dir = os.path.join(data_root, 'valid', str(cls))
    os.makedirs(target_dir)
    for sample in samples[nb_train:]:
        shutil.move(sample, target_dir)

    os.rmdir(cls_path)
