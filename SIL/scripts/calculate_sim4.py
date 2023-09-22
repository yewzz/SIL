import os
import argparse
import json

import models.weakly_graph.fusion as fusion
import models.clip as CLIP
import glob

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_caption_file', type=str,
                        default="/home/wangye/wangye2/wangye/video localization/MM2020/data/charades_sta/train.json")
    parser.add_argument('--out_file', type=str, default="new_train2.json")
    parser.add_argument('--cuda_device', default=0, type=int)
    opts = parser.parse_args()

    # 2d-map  proposal
    num_clips = 32
    start = np.reshape(np.repeat(np.arange(0, num_clips)[:, np.newaxis], axis=1,
                                 repeats=num_clips), [-1])
    end = np.reshape(np.repeat(np.arange(1, num_clips + 1)[np.newaxis, :], axis=0,
                               repeats=num_clips), [-1])
    props = np.stack([start, end], -1)

    # predefined proposals
    idx = props[:, 0] < props[:, 1]
    props = props[idx]

    # predefined proposals graph
    iou_predefined = True
    print('iou_predefined graph', iou_predefined)

    props_torch = torch.from_numpy(props)

    model, preprocess = CLIP.load("ViT-B/32", device="cuda", jit=True)
    ref_caps = json.load(open(opts.ref_caption_file))
    uniq_sents = set()
    feature_path = "/home/wangye/wangye2/wangye/Charades_v1_rgb/"

    for idx, data in enumerate(ref_caps):
        id, duration, timestamp, sents = data[0], data[1], data[2], data[3]
        new_path = feature_path + id + "/*"
        props = props_torch.float() * duration / num_clips
        props = props.long()

        cnt = 0
        images = []
        for f in sorted(glob.iglob(new_path)):
            if cnt % 24 == 0:  # 4 frame per second
                images.append(preprocess(Image.open(f)))
            cnt = cnt + 1
        images = torch.stack(images, 0)  # [536, 3, 320, 213]
        images = images.cuda()

        # preprocess text
        arr = []
        arr_old = []
        arr_old.append(sents)
        sents = list(sents)
        #sents.insert(0, 'a video clip where')
        sents = ''.join(sents)
        arr.append(sents)
        text_input = CLIP.tokenize(arr).cuda()
        text_input_old = CLIP.tokenize(arr_old).cuda()
        with torch.no_grad():
            images_features = []
            for i in range(len(images)):
                images_feature = model.encode_image(images[i].unsqueeze(0))
                images_feature /= images_feature.norm(dim=-1, keepdim=True)
                images_features.append(images_feature)
            images_features = torch.stack(images_features)
            map_features = []
            for i in range(len(props)):
                if props[i, 0] + 1 < props[i, 1] :
                    z = images_features[props[i, 0]:props[i, 1]]
                    z = z.mean(0) # z.max(0)[0]
                    #z = z / z.norm(dim=-1, keepdim=True)
                    map_features.append(z)
                else:
                    map_features.append(images_features[props[i, 0]])
            map_features = torch.stack(map_features)
            map_features = map_features / map_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_old = model.encode_text(text_input_old)
            text_features_old /= text_features.norm(dim=-1, keepdim=True)
            similarity = (map_features @ text_features.T.unsqueeze(0)).squeeze(-1).squeeze(
                -1)  # .softmax(dim=0) # [b,1,512] [1,512,1]
            x_max, x_min = similarity.max(dim=-1, keepdim=True)[0], similarity.min(dim=-1, keepdim=True)[0]
            for i in range(len(similarity)):
                if props[i, 1] - props[i, 0]>=5:
                    if (similarity[i] - x_min) / (x_max - x_min)>=0.7:
                        similarity[i] = similarity[i] * 1.2
                    elif (similarity[i] - x_min) / (x_max - x_min)>=0.7:
                        similarity[i] = similarity[i] * 0.8
            x_max, x_min = similarity.max(dim=-1, keepdim=True)[0], similarity.min(dim=-1, keepdim=True)[0]
            x = (similarity - x_min + 1e-10) / (x_max - x_min + 1e-10)
            pos = torch.argsort(similarity, descending=True)[:10]
            top_sim = x[pos]
            top_timestamp = props[pos] / num_clips * duration
            # gap = top_timestamp[:, 1] - top_timestamp[:, 0]
            #
            # longest = torch.argmax(gap)
            # target = top_timestamp[longest,:]
            similarity2 = (map_features @ text_features_old.T.unsqueeze(0)).squeeze(-1).squeeze(
                -1)  # .softmax(dim=0) # [b,1,512] [1,512,1]

            x_max2, x_min2 = similarity2.max(dim=-1, keepdim=True)[0], similarity2.min(dim=-1, keepdim=True)[0]
            x2 = (similarity2 - x_min2 + 1e-10) / (x_max2 - x_min2 + 1e-10)
            true_pos = torch.tensor(timestamp) / duration * 32
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