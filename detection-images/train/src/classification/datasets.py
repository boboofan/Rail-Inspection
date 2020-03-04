import numpy as np
import torch
import torchvision

import cv2


def train_transform(arr):
    arr = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_AREA)

    # padding and cropping
    arr = np.pad(arr, [[4, 4], [4, 4], [0, 0]], mode='constant')
    xx, yy = np.random.randint(5), np.random.randint(5)
    arr = arr[yy:yy + 64, xx:xx + 64, :]

    # random horizontal flipping
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1, :]

    arr = np.transpose(arr, [2, 0, 1]).astype(np.float32)
    arr = torch.from_numpy(arr)

    return arr


def valid_transform(arr):
    arr = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_AREA)
    arr = np.reshape(arr, (1, 64, 64, 3))
    arr = arr.astype(np.float32)
    arr = torch.from_numpy(arr)

    return arr


# def target_transform(target):
#     mapping = {
#          1: 0,
#          2: 1,
#          3: 2,
#          4: 3,
#         10: 4,
#         11: 5,
#         12: 6,
#         13: 7
#     }
#     return mapping[target]


class TrackClassification(torchvision.datasets.DatasetFolder):

    def __init__(self, root, train=True):
        super().__init__(
            root,
            np.load,
            extensions='npy',
            transform=train_transform if train else valid_transform,
            #             target_transform=target_transform
        )

    #     def __getitem__(self, index):
    #         img, target = super().__getitem__(index)
    #         print(target)
    #         return img, target

    def label2target(self, label):
        mapping = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 10,
            5: 11,
            6: 12,
            7: 13
        }
        return mapping[label]
