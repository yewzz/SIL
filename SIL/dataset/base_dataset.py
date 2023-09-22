import numpy as np
import torch
from gensim.utils import tokenize
from torch.utils.data import Dataset
import json
from utils import load_json
import nltk

def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou

class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, is_training, dataset='ActivityNetGraph', **kwargs):
        self.vocab = vocab  # 0:PAD, 1:UNK
        #_, self.preprocess = CLIP.load("ViT-B/32", device="cuda", jit=False)
        self.args = args
        self.is_training = is_training
        self.data = load_json(data_path)
        self.data_path = data_path
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.target_stride = args['target_stride']
        self.dataset = dataset
        if dataset == 'ActivityNetGraph':
            self.vid2num = load_json('/home/wangye/wangye2/wangye/video localization/ABIN/data/ActivityNet/vid.json')
            self.num2vid = {n: vid for vid, n in self.vid2num.items()}
        self.glove = np.load(open('/home/wangye/wangye2/wangye/video localization/ABIN/data/glove_emb', 'rb')).astype(np.float32)

        self.keep_vocab = dict()
        for w in vocab:
            self.keep_vocab[w] = self.vocab_size

        self.k = 5
        self.num_samples = len(self.data)

    @property
    def vocab_size(self):
        return len(self.keep_vocab) + 1
        
    def load_data(self, data):
        self.data = data

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _load_audio_features(self, vid):
        raise NotImplementedError

    def _load_teacher_frame_features(self, vid):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data[0])==4:
            vid, duration, timestamps, sentence,  = self.data[index]
            ret = {'raw': [vid, duration, timestamps, sentence]}
        else:
            vid, duration, timestamps, sentence, audio_id, spk_id = self.data[index]
            ret = {'raw': [vid, duration, timestamps, sentence, audio_id, spk_id]}

        duration = float(duration)

        words = [w.lower() for w in nltk.word_tokenize(sentence)
                 if w.lower() in self.vocab and w not in {'.', '\'\'', '\''}][:self.args['max_num_words']]
        weights = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            if word in self.vocab:
                if 'NN' in tag:
                    weights.append(2)
                elif 'VB' in tag:
                    weights.append(2)
                elif 'JJ' in tag or 'RB' in tag:
                    weights.append(2)
                else:
                    weights.append(1)
                # words.append(word)

        if len(words)==0:
            words.append('<PAD>')
            weights.append(0)

        if self.dataset == 'ActivityNetGraph':
            words_id = [self.keep_vocab[w] for w in words]
            words_feat = [self.glove[self.vocab[w]] for w in words]
        else:
            words_feat = [self.vocab[w] for w in words]

        if len(self.data[0])==4 or self.dataset=='CharadesSTA':
            frames_feat = self._load_frame_features(vid)  # [1:,:]

        else:
            frames_feat = self._load_frame_features(vid)  # [1:,:]
            audio_feat, audio_feat_wv = self._load_audio_features(audio_id)#.squeeze(0) # squeeze for wav2vec2.0
            audio_feat = audio_feat[::4, :]
            audio_feat_wv = audio_feat_wv[::4].reshape(-1)
        num_clips = self.max_num_frames // self.target_stride

        props = self.props_torch.float() * duration / num_clips
        gts = torch.tensor([timestamps[0], timestamps[1]]).unsqueeze(0).expand(props.size(0), -1).float()
        map_gt = calculate_IoU_batch((props[:, 0], props[:, 1]), (gts[:, 0], gts[:, 1]))


        reg_gt = self.props_torch[torch.argmax(map_gt)]
        frame_gt = torch.zeros(num_clips).float()
        frame_gt[reg_gt[0]:reg_gt[1] - 1] = 1.0


        if self.is_training :
            ret.update({
                'frames_feat': frames_feat,
                'words_feat': words_feat,
                'map_gt': map_gt,
                'reg_gt': reg_gt,
                'frame_gt': frame_gt,
                'audio_feat': audio_feat,
                'audio_feat_wv': audio_feat_wv,
                'spk_id': torch.tensor(int(spk_id)),
                'words_id': words_id,
                'weights': weights,
                'notnoise':torch.tensor(1),
            })
        else:
            ret.update({
                'frames_feat': frames_feat,
                'words_feat': words_feat,
                'words_id': words_id,
                'map_gt': map_gt,
                'reg_gt': reg_gt,
                'frame_gt': frame_gt,
                'audio_feat': audio_feat,
                'audio_feat_wv': audio_feat_wv,
                'weights': weights,
                'spk_id': torch.tensor(int(spk_id)),
                'notnoise': torch.tensor(1),
            })

        return ret


def build_collate_data(max_num_frames, max_num_words, max_num_audio, frame_dim, word_dim, audio_dim, props=None, props_graph=None, args=None, is_training=False):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }
        # vid_gt = [batch['raw'][i][0] for i in range(bsz)]
        frames_len = []
        words_len = []
        audio_len = []
        audio_len_wv = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))
            words_len.append(min(len(sample['words_id']), max_num_words))
            audio_len.append(min(len(sample['audio_feat']), 512))
            audio_len_wv.append(min(len(sample['audio_feat_wv']), 60000))

        # frames_feat = np.zeros([bsz, max(frames_len), frame_dim]).astype(np.float32)
        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)
        words_feat = np.zeros([bsz, max_num_words, word_dim]).astype(np.float32)
        audio_feat = np.zeros([bsz, 512, 80]).astype(np.float32)
        audio_feat_wv = np.zeros([bsz, 60000]).astype(np.float32)
        weights = np.zeros([bsz, max_num_words]).astype(np.float32)
        words_id = np.zeros([bsz, max_num_words]).astype(np.int64)

        map_gt = []
        reg_gt = []
        frame_gt = []
        spk_id = []
        notnoise = []

        rep = []
        for i, sample in enumerate(samples):
            # print(len(sample['words_feat']))
            # print(words_feat.shape)
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]

            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]


            rep.append(np.mean(sample['words_feat'], axis=0))
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)

            keep_audio = min(len(sample['audio_feat']), audio_feat.shape[1])
            audio_feat[i, :keep_audio] = sample['audio_feat'][:keep_audio]

            keep_audio_wv = min(len(sample['audio_feat_wv']), audio_feat_wv.shape[1])
            audio_feat_wv[i, :keep_audio_wv] = sample['audio_feat_wv'][:keep_audio_wv]

            if len(sample['frames_feat']) < frames_feat.shape[1] and False:
                frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            else:
                keep_idx = np.arange(0, frames_feat.shape[1] + 1) / frames_feat.shape[1] * len(sample['frames_feat'])
                keep_idx = np.round(keep_idx).astype(np.int64)
                keep_idx[keep_idx >= len(sample['frames_feat'])] = len(sample['frames_feat']) - 1
                frames_len[i] = frames_feat.shape[1]

                map_gt.append(sample['map_gt'])
                reg_gt.append(sample['reg_gt'])
                frame_gt.append(sample['frame_gt'])
                spk_id.append(sample['spk_id'])
                notnoise.append(sample['notnoise'])


                for j in range(frames_feat.shape[1]):
                    s, e = keep_idx[j], keep_idx[j + 1]
                    assert s <= e
                    if s == e:
                        frames_feat[i, j] = sample['frames_feat'][s]
                    else:
                        frames_feat[i, j] = sample['frames_feat'][s:e].mean(axis=0)

        neg = []
        rev = list(reversed(range(0, bsz)))
        for i in range(bsz):
            # print(i, np.argsort(dist[i]))
            # idx = np.random.choice(np.argsort(-dist[i])[:32])
            # print(dist[i])

            idx = rev[i]
            # idx = np.random.choice(bsz, p=dist[i])
            # print(idx, batch)
            try:
                if args['erase']:
                    while batch['raw'][i][0] == batch['raw'][idx][0]:  # or \
                        # batch['raw'][i][4][0] == batch['raw'][idx][4][0] or batch['raw'][i][4][1] == batch['raw'][idx][4][1]:
                        idx -= 1
                else:
                    while batch['raw'][i][0] == batch['raw'][idx][0]:
                        idx -= 1
            except:
                idx = np.random.choice(bsz)
                #if is_training:
                #    print('no-neg', idx)

            neg.append(idx)
        neg = np.asarray(neg)
        # print(neg)

        if is_training:
            batch.update({
                'net_input': {
                    'frames_feat': torch.from_numpy(frames_feat),
                    'frames_len': torch.from_numpy(np.asarray(frames_len)),
                    'words_feat': torch.from_numpy(words_feat),
                    'audio_feat': torch.from_numpy(audio_feat),
                    'audio_feat_wv': torch.from_numpy(audio_feat_wv),
                    'words_len': torch.from_numpy(np.asarray(words_len)),
                    'audio_len': torch.from_numpy(np.asarray(audio_len)),
                    'audio_len_wv': torch.from_numpy(np.asarray(audio_len_wv)).long(),
                    'map_gt': torch.stack(map_gt, 0),
                    'reg_gt': torch.stack(reg_gt, 0),
                    'frame_gt': torch.stack(frame_gt, 0),
                    'props': props.unsqueeze(0),
                    'props_graph': props_graph.unsqueeze(0),
                    'neg': neg,
                    'spk_id':torch.stack(spk_id, 0),
                    'weights': torch.from_numpy(weights),
                    'words_id': torch.from_numpy(words_id),
                    'notnoise':torch.stack(notnoise, 0),
                }
            })
        else:
            batch.update({
                'net_input': {
                    'frames_feat': torch.from_numpy(frames_feat),
                    'frames_len': torch.from_numpy(np.asarray(frames_len)),
                    'words_feat': torch.from_numpy(words_feat),
                    'audio_feat': torch.from_numpy(audio_feat),
                    'audio_feat_wv': torch.from_numpy(audio_feat_wv),
                    #'noisy_feat': torch.from_numpy(noise_feat),
                    'words_len': torch.from_numpy(np.asarray(words_len)),
                    'audio_len': torch.from_numpy(np.asarray(audio_len)),
                    'audio_len_wv': torch.from_numpy(np.asarray(audio_len_wv)),
                    'map_gt': torch.stack(map_gt, 0),
                    'reg_gt': torch.stack(reg_gt, 0),
                    'frame_gt': torch.stack(frame_gt, 0),
                    'props': props.unsqueeze(0),
                    'props_graph': props_graph.unsqueeze(0),
                    'neg': neg,
                    'spk_id':torch.stack(spk_id, 0),
                    'weights': torch.from_numpy(weights),
                    'words_id': torch.from_numpy(words_id),

                }
            })

        return batch

    return collate_data
