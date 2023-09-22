import os
import argparse
import json


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ref_caption_file',type=str, default="/home1/lihaoyuan/video localization/ABIN/data/ActivityNet/test_data.json")
  parser.add_argument('--ref_caption_file2', type=str, default="/home1/lihaoyuan/wangye/TIP2021-erase/ref_captions_audio.json")
  parser.add_argument('--out_file', type=str, default="test_data_audio.json")
  parser.add_argument('--cuda_device', default=0, type=int)
  opts = parser.parse_args()

  '''
  ref_caps = json.load(open(opts.ref_caption_file))
  ref_caps2 = json.load(open(opts.ref_caption_file2))
  uniq_sents = {}
  for id, content in enumerate(ref_caps2):
      sent = ref_caps2[content]
      uniq_sents[sent[0].replace(" ","")] = content
  uniq_sents = json.dumps(uniq_sents)
  f = open('ref_captions_audio.json', 'w')
  f.write(uniq_sents)
  f.close()
  #print('unique sents', len(uniq_sents))
  '''


  ref_caps = json.load(open(opts.ref_caption_file))
  ref_caps2 = json.load(open(opts.ref_caption_file2))
  uniq_sents = {}
  for id, content in enumerate(ref_caps):
      sent = content[3].replace(" ", "").replace("\n","")
      ref_cap = ref_caps2[sent]
      ref_caps[id].append(ref_cap)
      # ref_caps[idx].append(similarity_self.tolist())
      #print(id)


  with open(opts.out_file, 'w') as f:
      json.dump(ref_caps, f)



if __name__ == '__main__':
  main()