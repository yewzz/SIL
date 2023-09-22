import os
import argparse
import json

import models.weakly_graph.fusion as fusion
import models.clip as CLIP
import glob

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from gensim.corpora.wikicorpus import tokenize
import nltk
from gensim.models import KeyedVectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_caption_file', type=str,
                        default="/home/wangye/wangye2/wangye/video localization/MM2020/data/charades_sta/train.json")
    parser.add_argument('--out_file', type=str, default="new_train_txt.json")
    parser.add_argument('--cuda_device', default=0, type=int)
    opts = parser.parse_args()

    model, preprocess = CLIP.load("ViT-B/32", device="cuda", jit=True)
    ref_caps = json.load(open(opts.ref_caption_file))
    uniq_sents = set()
    feature_path = "/home/wangye/wangye2/wangye/Charades_v1_rgb/"

    vocab_path = "/home/wangye/wangye2/wangye/video localization/MM2020/data/charades_sta/glove_model.bin"
    vocab = KeyedVectors.load_word2vec_format(vocab_path, binary=True)


    for idx, data in enumerate(ref_caps):
        id, duration, timestamp, sents = data[0], data[1], data[2], data[3]
        new_path = feature_path + id + "/*"
        cnt = 0
        odd = 0
        images = []
        images2 = []
        path_file_number = glob.glob(pathname=new_path)  # 获取当前文件夹下个数
        sample_num = len(path_file_number) // 64
        sample_num_2 = sample_num // 2
        #arr = torch.linspace(0,len(path_file_number)-1,64).int()
        arr = torch.linspace(0, len(path_file_number)-1, 64).int()
        a = {}
        b = {}
        for x in arr:
            a[x.item()] = True
        for f in sorted(glob.iglob(new_path)):
            if a.__contains__(cnt) :
                if odd == 0:
                    images.append(preprocess(Image.open(f)))
                    odd=1
                else:
                    images2.append(preprocess(Image.open(f)))
                    odd=0
            cnt = cnt + 1
        images = torch.stack(images, 0)  # [536, 3, 320, 213]
        images = images.cuda()
        images2 = torch.stack(images2, 0)
        images2 = images2.cuda()


        # preprocess text
        arr = []
        words = [w.lower() for w in nltk.word_tokenize(sents)
                 if w.lower() in vocab and w not in {'.', '\'\'', '\''}][:25]
        for w in words:
            arr.append(w)
        text_input = CLIP.tokenize(arr).cuda()
        images_features = []
        with torch.no_grad():
            for i in range(len(images)):
                images_feature = model.encode_image(images[i].unsqueeze(0))
                images_feature /= images_feature.norm(dim=-1, keepdim=True)
                if i < len(images2):
                    images_feature2 = model.encode_image(images2[i].unsqueeze(0))
                    images_feature2 /= images_feature2.norm(dim=-1, keepdim=True)
                    images_feature = (images_feature + images_feature2) / 2
                    images_feature /= images_feature.norm(dim=-1, keepdim=True)
                images_features.append(images_feature)
            images_features = torch.stack(images_features).mean(dim=0) # [1, 1, 512]
            clip_features = images_features
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True) # [6, 512]
            similarity = (clip_features @ text_features.T.unsqueeze(0)).squeeze(0).squeeze(            # [1, 1, 6]
                0)  # .softmax(dim=0) # [b,1,512] [1,512,1]

            x_max, x_min = similarity.max(dim=-1, keepdim=True)[0], similarity.min(dim=-1, keepdim=True)[0]
            x = (similarity - x_min + 1e-10) / (x_max - x_min + 1e-10)
            ref_caps[idx].append(similarity.tolist())
            print(idx)

    with open(opts.out_file, 'w') as f:
        json.dump(ref_caps, f)
    print("ok")


'''
  outs = {}
  if os.path.exists(opts.out_file):
    outs = json.load(open(opts.out_file))
  for i, sent in enumerate(uniq_sents):
    if sent in outs:
      continue
    try:
      out = predictor.predict_tokenized(sent.split())
      # print("zzz")
    except KeyboardInterrupt:
      break
    except:
      continue
    outs[sent] = out
    if i % 1000 == 0:
      print('finish %d / %d = %.2f%%' % (i, len(uniq_sents), i / len(uniq_sents) * 100))

  with open(opts.out_file, 'w') as f:
    json.dump(outs, f)
    '''
if __name__ == '__main__':
    main()