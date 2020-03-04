import os
import gc
import shutil
import time

from io import StringIO
from socketserver import StreamRequestHandler, ThreadingTCPServer

import numpy as np
import pandas as pd
import torch
import cv2

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms

import sys
import os
sys.path.append(os.path.dirname(__file__))

from model import TrackDetectionModel

SCALE = 10

xwindow = 5000
xstride = 2000


class Track(Dataset):

    def __init__(self, filepath_or_buffer, args):

        self.args = args

        self.img = cv2.imread(filepath_or_buffer)
        self.img = np.array(self.img, dtype=np.float32)

    def __getitem__(self, index):
        return torch.from_numpy(np.transpose(self.img, axes=[2, 0, 1])).to(torch.half), {'x': torch.tensor(0),
                                                                                         'y': torch.tensor(0)}

    def __len__(self):
        return 1

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def name2label(self, name):
        mapping = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            10: 5,
            11: 6,
            12: 7,
            13: 8
        }
        return mapping[name]

    def label2name(self, label):
        mapping = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 10,
            6: 11,
            7: 12,
            8: 13
        }
        return mapping[label]

    def process(self, model):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        data = DataLoader(self,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          #         pin_memory=True,
                          collate_fn=self.collate_fn
                          )

        boxes = []
        scores = []
        labels = []

        for inputs, info in tqdm(data):

            inputs = [inp for inp in inputs if inp is not None]
            if len(inputs) == 0:
                continue

            inputs = [inp.to(device) for inp in inputs]

            with torch.no_grad():
                outputs = model(inputs)

            for inf, outp in zip(info, outputs):
                bs = outp['boxes'].to(torch.float)
                boxes.extend(bs)
                labels.extend(outp['labels'].to(torch.int))
                scores.extend(outp['scores'].to(torch.float))

        if not boxes:
            return {}

        boxes = torch.stack(boxes)
        scores = torch.stack(scores)
        labels = torch.stack(labels)

        ids = nms(boxes, scores, iou_threshold=self.args.box_score_thresh).cpu().numpy()
        return {
            'boxes': boxes[ids].cpu().numpy(),
            'labels': labels[ids].cpu().numpy(),
            'scores': scores[ids].cpu().numpy()
        }

    def postprocess(self, pred):
        if len(pred)==0:
            return pd.DataFrame()
        pred['labels'] = [self.label2name(int(label)) for label in pred['labels']]

        n = len(pred['boxes'])
        res = np.zeros((n, 6))
        res[:, :4] = pred['boxes']
        res[:, 4] = pred['labels']
        res[:, 5] = pred['scores']
        res = pd.DataFrame(res)
        res.sort_values(0, axis=0, inplace=True)

        return res


def main(args):
    model = TrackDetectionModel(
        box_score_thresh=args.box_score_thresh,
        box_nms_thresh=args.box_nms_thresh
    )
    model.load_state_dict(torch.load('checkpoint.pth')['model'])
    model.half().eval()

    # if args.online:
    #
    #     class PredHandler(StreamRequestHandler):
    #
    #         def handle(self):
    #             print('[INFO] Conneted from {}.'.format(self.client_address))
    #
    #             blen = int(str(self.rfile.readline(), 'utf-8'))
    #             buffer = self.rfile.read(blen)
    #             text = str(buffer, encoding='utf-8')
    #
    #             data = Track(StringIO(text), args)
    #             pred = data.process(model)
    #             res = data.postprocess(pred)
    #
    #             res = bytes(res.to_csv(index=False, header=False), encoding='utf-8')
    #             alen = bytes(str(len(res)) + '\n', encoding='utf-8')
    #             self.wfile.write(alen + res)
    #
    #     #                 print('[INFO] Disconneted from {}.'.format(self.client_address))
    #
    #     print('[INFO] Listening to 127.0.0.1:{}'.format(args.port))
    #     server = ThreadingTCPServer(
    #         ('127.0.0.1', args.port),
    #         PredHandler
    #     )
    #     server.serve_forever()

    while True:
        dir_list = os.listdir(args.input_dir)
        for dir in dir_list:
            dir_path = os.path.join(args.input_dir, dir)

            array_files = os.listdir(dir_path)
            if '1.txt' in array_files:
                continue
            
            end=True if '0.txt' in array_files else False

            cache_file_path = os.path.join(dir_path, 'cache.npy')
            if not os.path.exists(cache_file_path):
                cache = np.array([], dtype=str)
            else:
                cache = np.load(cache_file_path)

            array_files = [
                file
                for file in array_files
                if file.endswith('.png') and file not in cache
            ]

            for afile in array_files:
                cache = np.append(cache, afile)
                print(f'[INFO] processing {afile}')

                data = Track(os.path.join(dir_path, afile), args)
                pred = data.process(model)
                res = data.postprocess(pred)

                del data
                gc.collect()
   
                lfile = os.path.join(args.output_dir, dir + '.txt')
                if not os.path.exists(lfile):
                    label = pd.DataFrame()
                else:
                    try:
                        label = pd.read_csv(lfile, header=None)
                    except:
                        label = pd.DataFrame()

                    if len(res)>0:
                        label = label.append(pd.Series([afile, len(res)]), ignore_index=True)
                        label = label.append(res, ignore_index=True)

                label.to_csv(lfile, header=False, index=False)
            np.save(cache_file_path, cache)

            if end:
                flag=open(os.path.join(dir_path, '1.txt'),'w')
                flag.close()

        time.sleep(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predicting')

    parser.add_argument('-i', '--input-dir', default='in', help='path to input data')
    parser.add_argument('-o', '--output-dir', default='out', help='path where to save')

    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-n', '--num-workers', default=8, type=int, help='number of workers to load data')

    parser.add_argument('--box-score-thresh', default=0.05, type=float)
    parser.add_argument('--box-nms-thresh', default=0.5, type=float)

    parser.add_argument('--online', action="store_true", help='online predicting')
    parser.add_argument('--port', default=28889, type=int, help='TCP port where socket is open')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    main(args)
