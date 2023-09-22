import argparse
import json
import os

import librosa
import numpy as np
import soundfile
import random
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt

def awgn(x, snr, out='signal', method='vectorized', axis=0):

    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)

    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))

    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)

    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    # Signal power, in dB
    Psdb = 10 * np.log10(Ps)

    # Noise level necessary
    Pn = Psdb - snr

    # Noise vector (or matrix)
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)

    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n

def add_noise(audio_path, out_path, percent=0.2, sr=16000):
    src, sr = librosa.load(audio_path, sr=sr)
    snr = 0 # 1， 25
    waveData = awgn(src, snr, out='signal', method='vectorized', axis=0)
    random_values = np.random.rand(len(src))
    src = src + percent * random_values
    soundfile.write(out_path, waveData, sr)

def signal_by_db(x1, x2, snr, handle_method=None):
    from numpy.linalg import norm
    from math import sqrt
    import math

    #
    # target_clean_rms = 0.020
    # x1 = x1 - np.mean(x1)
    # stand_rms = np.sqrt(np.mean(x1))
    #
    # x1 = (target_clean_rms / stand_rms) * x1
    # stand_rms = np.sqrt(np.mean(x1))
    #
    # x2 = x2 - np.mean(x2)
    #
    # x2 = stand_rms / np.sqrt(np.mean(x2)) * x2

    # x1 = x1.astype(np.int16)
    # x2 = x2.astype(np.int16)
    x2= x2[:250000]
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    if l1 < l2:
        handle_method='cut'
    else:
        handle_method='append'

    if l1 != l2:
        if handle_method == 'cut':
            ll = min(l1, l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            ll = max(l1, l2)

            if l2 < ll:
                x2_total = []
                for i in range(int(l1 // l2)*3):
                    x2_total.append(x2)
                x2_total = np.hstack(x2_total)
                #    x2 = np.append(x2, x2[:l2])

                ll2 = x1.shape[0]

                x2 = x2_total[:ll2]

    x2 = x2 / norm(x2) * norm(x1) / (10.0 ** ( 0.05*snr))
    # x2 = math.sqrt(np.sum(np.abs(x1) ** 2)) / math.sqrt((np.sum(np.abs(x2) ** 2)) * (10 ** snr)) * x2
    # x2 = np.sqrt(np.sum(np.abs(x1)**2))/np.sqrt((np.sum(np.abs(x2)**2))*(10**snr))*x2
    mix = x1 + x2

    return mix, x2


def get_all_noise_files(num_noise_files=10):

    root_dir = '/home/wangye/wangye2/wangye/Nonspeech'
    ambience_files = ['{}/{}'.format(root_dir,i) for i in os.listdir(root_dir) ] # Ambience
    random.shuffle(ambience_files)
    files = {}
    files[0] = ambience_files[:] #:num_noise_files
    return files

def get_noise_files(all_noise_files,noise_class_ids=[0]):
    noise_files = []
    for c in noise_class_ids:
        noise_files += all_noise_files[c]
    random.shuffle(noise_files)
    return noise_files



if __name__ == "__main__":
    # test_json = json.load(open("/home/wangye/wangye2/wangye/video localization/ABIN/data/ActivityNet/test_data_audio_spk.json"))
    # uniq_sents = set()
    # for id, sents in enumerate(test_json):
    #     # for sent in sents:
    #     speechid = sents[4]
    #     uniq_sents.add(speechid)
    # uniq_sents = list(uniq_sents)
    noise_files = get_noise_files(get_all_noise_files())
    audio_dir = '/home/wangye/wangye2/wangye/wavs_16k/'
    out_dir = '/home/wangye/wangye2/wangye/wavs_16k_mixall/'
    cnt = 0
    #for file in uniq_sents:
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue
            #file = file + '.wav'
            audio_path = os.path.join("/home/wangye/wangye2/wangye/wavs_16k", file)
            audio_path = "/home/wangye/wangye2/wangye/wavs_16k_mixall/EE1UoJSNyms_train_2.wav"
            wav_input_16khz, rate = librosa.load(audio_path, sr=16000)
            fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
            librosa.display.waveplot(wav_input_16khz, sr=16000, x_axis='time', offset=0.0, ax=None)
            ax[0].set(title='Monophonic')
            ax[0].label_outer()
            plt.show()
            fig.savefig('audio_noise.png', transparent=True)



            nf = random.choice(noise_files)
            snr = np.random.randint(1, 11)
            wav_noise_16khz, fs = librosa.load(nf, sr=16000)

            mix, _ = signal_by_db(wav_input_16khz, wav_noise_16khz, snr)

            out_path = os.path.join(out_dir, file)
            soundfile.write(out_path, mix, 16000)
            cnt =  cnt + 1
            print(cnt)