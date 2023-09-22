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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ref_caption_file',type=str, default="/home/wangye/wangye2/wangye/video localization/MM2020/data/charades_sta/train.json")
  parser.add_argument('--out_file', type=str, default="new_train.json")
  parser.add_argument('--cuda_device', default=0, type=int)
  opts = parser.parse_args()

  model, preprocess = CLIP.load("ViT-B/32", device="cuda", jit=True)
  ref_caps = json.load(open(opts.ref_caption_file))
  uniq_sents = set()
  feature_path = "/home/wangye/wangye2/wangye/Charades_v1_rgb/"

  for idx, data in enumerate(ref_caps):
      id, duration, timestamp, sents = data[0], data[1], data[2], data[3]
      new_path = feature_path + id + "/*"
      cnt = 0
      images = []
      images2 = []
      for f in sorted(glob.iglob(new_path)):
          if cnt % 24 == 0:
              images.append(preprocess(Image.open(f)))
          elif cnt % 12 == 0:
              images2.append(preprocess(Image.open(f)))
          cnt = cnt + 1
      images = torch.stack(images, 0)  # [536, 3, 320, 213]
      images = images.cuda()
      images2 = torch.stack(images2, 0).cuda()

      # preprocess text
      arr = []
      arr.append(sents)
      text_input = CLIP.tokenize(arr).cuda()

      with torch.no_grad():
          logits_per_image, _ = model(images, text_input)
          #probs = logits_per_image.squeeze(-1).softmax(dim=-1)

          logits_per_image2, _ = model(images2, text_input)
          #probs2 = logits_per_image2.squeeze(-1).softmax(dim=-1)
          if len(logits_per_image)!=len(logits_per_image2):
              logits_per_image = logits_per_image[:-1]
          #final_prob = torch.max(logits_per_image, logits_per_image2)
          final_prob = logits_per_image # (logits_per_image + logits_per_image2)/2
          final_prob = final_prob.squeeze(-1)#.softmax(dim=-1)

          temperature = torch.ones([]) * np.log(1 / 0.07)
          logit_scale = temperature.exp()
          final_prob = final_prob * 2
          final_prob = final_prob.softmax(dim=-1)
          #x_max, x_min = final_prob.max(dim=-1, keepdim=True)[0], final_prob.min(dim=-1, keepdim=True)[0]
          #x = (final_prob - x_min + 1e-10) / (x_max - x_min + 1e-10)
          ref_caps[idx].append(final_prob.tolist())
          print("idx")

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