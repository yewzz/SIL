import os
import argparse
import json

from allennlp.predictors.predictor import Predictor


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ref_caption_file',type=str, default="/home/wangye/wangye2/wangye/video localization/AAAI-2020/data/didemo/train_data.json")
  parser.add_argument('--out_file', type=str, default="sent2srl_didemo.json")
  parser.add_argument('--cuda_device', default=0, type=int)
  opts = parser.parse_args()

  predictor = Predictor.from_path("/home/wangye/wangye2/wangye/video localization/data/model.tar.gz", cuda_device=opts.cuda_device)

  ref_caps = json.load(open(opts.ref_caption_file))
  uniq_sents = set()
  for id, sents in enumerate(ref_caps):
    # for sent in sents:
    if sents['description'][0] == ' ':
      uniq_sents.add(sents['description'][1:-1])
    uniq_sents.add(sents['description'][:-1])
  uniq_sents = list(uniq_sents)
  print('unique sents', len(uniq_sents))

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

if __name__ == '__main__':
  main()