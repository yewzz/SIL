import os
import argparse
import json
import glob

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ref_caption_file',type=str, default="/home1/lihaoyuan/wangye/TIP2021-erase/test_data_audio.json")
  parser.add_argument('--ref_caption_file2', type=str, default="/home1/lihaoyuan/wangye/id2index.json")
  parser.add_argument('--out_file', type=str, default="test_data_audio_spk.json")
  parser.add_argument('--cuda_device', default=0, type=int)
  opts = parser.parse_args()


  ref_caps = json.load(open(opts.ref_caption_file))
  ref_caps2 = json.load(open(opts.ref_caption_file2))
  ref_caps3 = json.load(open("/home1/lihaoyuan/wangye/TIP2021-erase/speaker2id.json"))
  uniq_sents = {}
  for idx, content in enumerate(ref_caps):
      aid = content[-1]
      #aid = "---" + aid
      id = ref_caps2[aid]
      value = int(id)
      if value >= 100:
        id = id.rjust(6, '0')
      else:
        id = id.rjust(5, '0')
        id = id + '.'
      # if id == '00089':
      #     print('??')
      sid = ref_caps3[id]
      ref_caps[idx].append(sid)
      #uniq_sents[sent[0].replace(" ","")] = content
  with open(opts.out_file, 'w') as f:
      json.dump(ref_caps, f)

  #print('unique sents', len(uniq_sents))

  '''
  new_path = '/home1/lihaoyuan/wangye/wavs/*'
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '3'

  s_v = {}
  cnt = 0
  for f in sorted(glob.iglob(new_path)):
      cnt = cnt + 1
      s_id = f[32:35]
      v_id = f[39:45]
      s_v[v_id] = s_id
      print(cnt)

  

  with open(opts.out_file, 'w') as f:
      json.dump(s_v, f)
  '''


if __name__ == '__main__':
  main()