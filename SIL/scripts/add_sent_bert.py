import os
import argparse
import json

import glob

import torch
from PIL import Image
import torchvision.transforms as transforms
from fairseq.utils import move_to_cuda

import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_caption_file', type=str,
                        default="/home/wangye/wangye2/wangye/TIP2021-erase/train_corpus.json")
    parser.add_argument('--out_file', type=str, default="train_corpus_sent.json")
    parser.add_argument('--cuda_device', default=2, type=int)
    opts = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from transformers import BertModel, BertConfig, BertTokenizer
    #model = SentenceTransformer('all-mpnet-base-v2').cuda()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained("bert-base-uncased",
                                        output_hidden_states=True)
    model = BertModel(config)#.cuda()
    ref_caps = json.load(open(opts.ref_caption_file))
    uniq_sents = set()
    # feature_path = "/home/wangye/wangye2/wangye/Charades_v1_rgb/"

    for idx, data in enumerate(ref_caps):
        dict_data = eval(data)
        id, duration, split_word, timestamp, sentence = dict_data['key'], dict_data['duration'], dict_data['word'], \
                                                        dict_data['timestamp'], dict_data['text']

        # new_path = feature_path + id + "/*"

        input = tokenizer(sentence, return_tensors="pt")#.cuda()
        # input = move_to_cuda(input)
        output = model(**input)
        #sent_bert = model.encode([sentence])

        layer_feat = output['hidden_states']
        # layer_feats = []
        # for i in range(len(layer_feat)):
        #     layer_feats.append(layer_feat[i].squeeze(0).tolist())
        dict_data['sentence'] = output['last_hidden_state'][0, 0, :].view(-1).tolist()
        ref_caps[idx] = json.dumps(dict_data)

        # np_arr = np.stack(layer_feats, 0)

        # out_file = '/home1/yeshangwei/wangye/data/layer-feat/' + id.split("/")[-1][:-5] + '.npy'
        # with open(out_file, 'wb') as f:
        #     np.save(f, np_arr)
        print(idx)

    with open(opts.out_file, 'w') as f:
        json.dump(ref_caps, f)
    print("ok")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()