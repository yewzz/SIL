import torch
import fairseq
import librosa
import glob
import numpy as np
from pathlib import Path

import soundfile as sf
new_path = '/home1/lihaoyuan/wangye/wavs_16k_noise/*'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def main():
    cnt = 0
    for f in sorted(glob.iglob(new_path)):
        cnt = cnt + 1
        if cnt < 67000:
            continue
        if cnt == 69000:
             break


        out_file = "/home1/lihaoyuan/wangye/TIP2021-erase/wav2vec_noise_np/" + f[39:-4] + ".npy"
        out_file_path = Path(out_file)
        if out_file_path.exists():
            continue
        cp_path = '/home1/lihaoyuan/wangye/TIP2021-erase/wav2vec_large.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        model = model[0].cuda(0)
        model.eval()

        wav_input_16khz, rate = librosa.load(f, sr=16000)
        #y, _ = librosa.effects.trim(wav_input_16khz, top_db=5)
        # wav_fn = f"{processed_dir}/wav_inputs/{basename}.wav"
        # y = y.write("./p225_004_2_L_2.wav", y, 16000)
        # sf.write('stereo_file1.wav', y, samplerate=16000)
        #print(librosa.get_duration(y), librosa.get_duration(wav_input_16khz))
        wav_input_16khz = torch.from_numpy(wav_input_16khz).view(1, -1).cuda(0)
        #wav_input_16khz = wav_input_16khz.expand(5, -1)
        # wav_input_16khz = torch.randn(1, 10000)
        z = model.feature_extractor(wav_input_16khz)
        c = model.feature_aggregator(z)


        np_arr = c.detach().cpu().numpy()
        with open(out_file, 'wb') as f:
            np.save(f, np_arr)
        print(f)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    x = '3.wav'
    a = x[:-4]
    main()