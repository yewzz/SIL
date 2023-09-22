import os

import h5py

from utils import iou
import numpy as np
import torch
from gensim.utils import tokenize
from torch.utils.data import Dataset
from bert_embedding import BertEmbedding
from gensim.models import KeyedVectors
import librosa
# import mxnet
import nltk

import warnings
# warnings.filterwarnings('once')
warnings.simplefilter('ignore', ResourceWarning)
warnings.simplefilter('ignore', UserWarning)

from utils import load_json, load_json_line
def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


from torch.utils.data import Dataset, DataLoader
class LibriSpeech(Dataset):
    def __init__(self, data_path, vocab_path, glove_sent_path, args, is_training=True, **kwargs):
        super().__init__()

        self.args = args
        self.is_training = is_training
        self.data = load_json_line(data_path)
        self.data_path = data_path
        self.ori_data = self.data
        self.bert_embedding = BertEmbedding(max_seq_length=50)

        self.glove_vocab = KeyedVectors.load_word2vec_format(vocab_path, binary=True)
        self.glove_keys = self.glove_vocab.key_to_index.keys()

        self.glove_keys_common = self.glove_vocab.index_to_key[:10000]

        self.glove_sent = np.load("lib_glove_feat.npy")
        self.glove_sent_len = np.load("lib_glove_len.npy")

        self.neg_k = 256

    @property
    def vocab_size(self):
        return len(self.glove_vocab.vectors) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dict_data = self.data[index]
        id, duration, sentence = dict_data['key'], dict_data['duration'], dict_data['sentence'] #
        duration = float(duration)

        weights = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)) :
            word = word.lower()
            if word in self.glove_keys  and word not in {'.', '\'\'', '\''}: #and self.glove_vocab.key_to_index[word] < 10000:
                if 'NN' in tag:
                    weights.append(2)
                elif 'VB' in tag:
                    weights.append(2)
                elif 'JJ' in tag or 'RB' in tag:
                    weights.append(2)
                else:
                    weights.append(1)

        # split_word = [w for w in split_word if w!='']
        words = [w.lower() for w in nltk.word_tokenize(sentence)
                 if w.lower() in self.glove_keys and w not in {'.', '\'\'', '\''}]
        word_id = [self.glove_vocab.key_to_index[w] for w in words]  # 1-vocab_size

        audio_feat = self._load_audio_features(id)
        audio_feat = audio_feat[::4, :]
        def audio2vector(audio_filename, dim_feature):

            import soundfile
            sig, _ = soundfile.read(audio_filename)
            sig = sig.reshape(sig.shape[0], 1)
            return  sig

        audio_feat_wv = audio2vector(id, None)
        audio_feat_wv = audio_feat_wv[::4, :].reshape(-1)
        ret = {'raw': [id, duration, sentence]}

        word_id2 = [w for w in word_id if w < 10000]
        words_feat = self.glove_vocab.vectors[word_id2]
        if len(words_feat) == 0:
            words_feat = self.glove_vocab.vectors[0].reshape(-1, 300)
            word_id2 = [0]
            weights = [1]

        if len(word_id2) == 0:
            word_id2 = [0]
            weights = [1]


        neg_idx = np.random.choice(len(self.data), self.neg_k, replace=True)
        neg_idx[neg_idx == index] = index + 1 if index + 1 < len(self.data) else index - 1
        neg_sent_feat = self.glove_sent[neg_idx]
        neg_sent_len = self.glove_sent_len[neg_idx]


        audio_word_label = np.ones([127]) * 10000
        audio_len = audio_feat.shape[0]
        word_len_all = max(len(words), 1)


        ret.update({
            'audio_feat': audio_feat,
            'audio_feat_wv': audio_feat_wv,
            'weights': weights,
            'words_id': word_id2,
            'words_id_all': word_id,
            'words_feat': words_feat,
            'idx': index,
            'word_len_all': word_len_all,
            'neg_sent_feat': neg_sent_feat,
            'neg_sent_len': neg_sent_len,
            'audio_word_label': audio_word_label,
        })


        return ret

    def _load_frame_features(self, vid):
        with h5py.File(os.path.join(self.args['feature_path'], '%s.h5' % vid), 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)

    def _load_audio_features(self, vid):
        vid = list(vid)
        vid[-4:] = 'npy'
        vid = ''.join(vid)

        return np.asarray(
            np.load(vid)).astype(
           np.float32)


    def collate_data(self, samples):
        return self.collate_fn(samples)


    def collate_fn(self, samples):

        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        words_len = []
        audio_len = []
        audio_len_wv = []
        words_id_len = []
        words_len_all = torch.from_numpy(np.asarray([sample['word_len_all'] for sample in samples])).long()
        neg_words_len = []


        for i, sample in enumerate(samples):
            words_len.append(min(len(sample['words_feat']), 50))
            neg_words_len.append(sample['neg_sent_len'])
            audio_len.append(min(len(sample['audio_feat']), 512))
            audio_len_wv.append(min(len(sample['audio_feat_wv']), 60000))
            words_id_len.append(min(len(sample['words_id']), 50))

        words_feat = np.zeros([bsz, 50, 300]).astype(np.float32)
        neg_words_feat = np.zeros([bsz, self.neg_k, 50, 300]).astype(np.float32)
        audio_feat = np.zeros([bsz, 512, 80]).astype(np.float32)
        audio_feat_wv = np.zeros([bsz, 60000]).astype(np.float32)
        words_id = np.ones([bsz, 50]).astype(np.int) * 10000 #.astype(n)
        words_id_all = np.zeros([bsz, 50]).astype(np.int)
        weights = np.zeros([bsz, 50]).astype(np.float32)

        for i, sample in enumerate(samples):
            keep_audio = min(len(sample['audio_feat']), audio_feat.shape[1])
            audio_feat[i, :keep_audio] = sample['audio_feat'][:keep_audio]

            keep_audio_wv = min(len(sample['audio_feat_wv']), audio_feat_wv.shape[1])
            audio_feat_wv[i, :keep_audio_wv] = sample['audio_feat_wv'][:keep_audio_wv]

            keep_word = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep_word] = sample['words_id'][:keep_word]
            tmp = np.exp(sample['weights'][:keep_word])
            weights[i, :keep_word] = tmp / np.sum(tmp)

            keep_word_all = min(len(sample['words_id_all']), words_id_all.shape[1])
            words_id_all[i, :keep_word_all] = sample['words_id_all'][:keep_word_all]

            neg_words_feat[i, :, :] = sample['neg_sent_feat'][:]

            words_feat[i, :keep_word] = sample['words_feat'][:keep_word]

        idx = torch.from_numpy(np.asarray([sample['idx'] for sample in samples])).long()
        audio_word_label = torch.from_numpy(np.asarray([sample['audio_word_label'] for sample in samples]))

        if True:
            batch.update({
                'net_input': {
                    # 'bert_sen_feat': torch.from_numpy(np.asarray([samples[i]['bert_sen_feat'] for i in range(bsz)])),
                    'words_feat': torch.from_numpy(words_feat),
                    'neg_words_feat': torch.from_numpy(neg_words_feat),
                    'audio_feat': torch.from_numpy(audio_feat),
                    'audio_feat_wv': torch.from_numpy(audio_feat_wv),
                    'words_len': torch.from_numpy(np.asarray(words_len)).long(),
                    'neg_words_len': torch.from_numpy(np.asarray(neg_words_len)).long(),
                    'words_len_all': words_len_all,
                    'words_id': torch.from_numpy(words_id),
                    'words_id_all': torch.from_numpy(words_id_all),
                    'words_id_len': torch.from_numpy(np.asarray(words_id_len)).long(),
                    'audio_len': torch.from_numpy(np.asarray(audio_len)).long(),
                    'audio_len_wv': torch.from_numpy(np.asarray(audio_len_wv)).long(),
                    'idx': idx,
                    'audio_word_label': audio_word_label.long(),
                    'weights': torch.from_numpy(weights),


                }
            })

        return batch



