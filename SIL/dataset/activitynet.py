import os

import h5py

from dataset.base_dataset import BaseDataset, build_collate_data
from utils import iou
import numpy as np
import torch
from gensim.utils import tokenize
from torch.utils.data import Dataset
# from bert_embedding import BertEmbedding
import librosa

from utils import load_json
def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


class ActivityNetGraph(BaseDataset):
    def __init__(self, data_path, vocab, args, is_training=False, **kwargs):
        super().__init__(data_path, vocab, args, is_training, **kwargs)

        self.num_clips = args['max_num_frames'] // args['target_stride']
        start = np.reshape(np.repeat(np.arange(0, self.num_clips)[:, np.newaxis], axis=1,
                                     repeats=self.num_clips), [-1])
        end = np.reshape(np.repeat(np.arange(1, self.num_clips + 1)[np.newaxis, :], axis=0,
                                   repeats=self.num_clips), [-1])
        self.props = np.stack([start, end], -1)

        # construct 2D map for proposals

        self.props = []
        tmp = [[1], [2], [2]]
        tmp[0].extend([1] * 15)
        tmp[1].extend([1] * 7)
        tmp[2].extend([1] * 7)
        acum_layers = 0
        stride = 1

        for scale_idx, strides in enumerate(tmp):
            for i, stride_i in enumerate(strides):
                stride = stride * stride_i
                keep = False
                if is_training:
                    if scale_idx == 0 and i in [7, 15]:
                        keep = True
                    elif scale_idx == 1 and (i in [3, 7]):
                        keep = True
                    elif scale_idx == 2 and (i in [3, 5, 7]):
                        keep = True
                else:
                    if scale_idx == 0 and i in [7, 15]:
                        keep = True
                    elif scale_idx == 1 and (i in [3, 7]):
                        keep = True
                    elif scale_idx == 2 and (i in [3, 5, 7]):
                        keep = True
                if not keep:
                    continue
                ori_s_idxs = list(range(0, self.num_clips - acum_layers - i * stride, stride))
                ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]

                self.props.append(np.stack([ori_s_idxs, ori_e_idxs], -1))

            acum_layers += stride * (len(strides) + 1)
        self.props = np.concatenate(self.props, 0)
        self.props[:, 1] += 1

        print('candidate proposals', self.props.shape)
        # predefined proposals graph
        iou_predefined = True
        print('iou_predefined graph', iou_predefined)
        if iou_predefined:

            props_iou = iou(self.props.tolist(), self.props.tolist())  # shape:(153,153)
            self.props_graph = np.zeros_like(props_iou).astype(np.int32)
            sort_idx = np.argsort(-props_iou, -1)
            #props_len = self.props[:, 1] - self.props[:, 0]
            for i in range(self.props.shape[0]):
                prop_len = self.props[i, 1] - self.props[i, 0]
                self.props_graph[i, sort_idx[i]] = 1
                low_idx = props_iou[i] < 0.2 # 0.2
                self.props_graph[i, low_idx] = 0

        else:
            num_props = self.props.shape[0]
            self.props_graph = np.zeros([num_props, num_props]).astype(np.int32)
            for i in range(num_props):
                for j in range(num_props):
                    if abs(self.props[i, 0] - self.props[j, 0]) <= 3 and \
                            abs(self.props[i, 1] - self.props[j, 1]) <= 3:
                        self.props_graph[i, j] = 1
            print(self.props_graph.sum(axis=-1))
            exit(0)

        self.props_torch = torch.from_numpy(self.props) 
        self.props_graph_torch = torch.from_numpy(self.props_graph)
        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'], args['max_num_audio'],
                                                 args['frame_dim'], args['word_dim'], args['audio_dim'],
                                                 self.props_torch, self.props_graph_torch, args, is_training)

        # for each prop in 2D map, generate corresponding map IoU
        props = self.props_torch.float()
        self.map_iou = []
        for i in range(len(self.props_torch)):
            timestamps = self.props_torch[i]
            gts = torch.tensor([timestamps[0], timestamps[1]]).unsqueeze(0).expand(props.size(0), -1).float()
            map_gt = calculate_IoU_batch((props[:, 0], props[:, 1]), (gts[:, 0], gts[:, 1]))
            self.map_iou.append(map_gt)
        self.map_iou = torch.stack(self.map_iou, 0)



    def _load_frame_features(self, vid):
        with h5py.File(os.path.join(self.args['feature_path'], '%s.h5' % vid), 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)


    def _load_audio_features(self, vid):
        mel = np.asarray(
            np.load(os.path.join('/home/wangye/wangye2/wangye/TIP2021-erase/mel_np_2022/', '%s.npy' % vid))).astype(
           np.float32)
        wav, cur_sr = librosa.load(os.path.join('/home/wangye/wangye2/wangye/wavs_16k/', '%s.wav' % vid), sr=16000)
        return mel, wav.astype(np.float32)

        # return np.asarray(
        #     np.load(os.path.join('/home/wangye/wangye2/wangye/TIP2021-erase/mel_np_2022_mixall/', '%s.npy' % vid))).astype(
        #     np.float32)

    def collate_data(self, samples):
        return self.collate_fn(samples)


class ActivityNet(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'])

    def _load_frame_features(self, vid):
        with h5py.File(os.path.join(self.args['feature_path'], '%s.h5' % vid), 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)

    def _load_frame_features_rgb(self, vid):
        with h5py.File(os.path.join(self.args['feature_path'], '%s.h5' % vid), 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)



    def collate_data(self, samples):
        return self.collate_fn(samples)
