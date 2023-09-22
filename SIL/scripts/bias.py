import os
import argparse
import json
import numpy as np

def main():
  parser = argparse.ArgumentParser()
  #parser.add_argument('--ref_caption_file',type=str, default="/home1/lihaoyuan/video localization/ABIN/data/ActivityNet/test_data.json")
  parser.add_argument('--ref_caption_file', type=str, default="/home1/lihaoyuan/wangye/TIP2021-erase/spk_res_contrast")
  parser.add_argument('--out_file', type=str, default="??.json")
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
  #ref_caps2 = json.load(open(opts.ref_caption_file2))
  uniq_sents = {}
  # data = {}
  # for i in range(0, 110):
  #     data[i] = []
  # for id, content in enumerate(ref_caps):
  #     vid, duration, timestamp, sent, aid, spkid = content[0], content[1], content[2], content[3], content[4], content[5]
  #     spkid = int(spkid)
  #     start_time, end_time = timestamp[0], timestamp[1]
  #     start, end = (start_time / duration) * 32, (end_time / duration) * 32
  #     # if data[spkid] == None:
  #     #   data[spkid] = []
  #     #   data[spkid] = data[spkid].append([start, end])
  #     # else:
  #     data[spkid].append([start, end])
  #     #data.append(end)
  #
  # for i in data:
  #     if len(data[i])!=0:
  #       data[i] = np.stack(data[i], 0)
      #data = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)


  import matplotlib.pyplot as pl

  import scipy.stats as st
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.stats import gaussian_kde
  from matplotlib.colors import LogNorm

  for content in ref_caps:
      if len(ref_caps[content])!=0:
          data_ = ref_caps[content]
          data_ = np.stack(data_, 0)
          x = data_[:, 0]

          y = data_[:, 1]

          xmin = 0
          xmax = 64
          ymin = 0
          ymax = 64



          X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
          positions = np.vstack([X.ravel(), Y.ravel()])
          values = np.vstack([x, y])
          kernel = gaussian_kde(values)
          Z = np.reshape(kernel(positions).T, X.shape)


          import matplotlib.pyplot as plt
          fig, ax = plt.subplots()
          ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent = [xmin, xmax, ymin, ymax])
          ax.plot(x, y, 'k.', markersize=2)
          ax.set_xlim([xmin, xmax])
          ax.set_ylim([ymin, ymax])
          plt.show()
  with open(opts.out_file, 'w') as f:
      json.dump(ref_caps, f)



if __name__ == '__main__':
  main()