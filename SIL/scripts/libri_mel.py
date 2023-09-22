import librosa
import time
import numpy as np
from python_speech_features import mfcc, logfbank
import soundfile

def load_wavfile(wavfile):
    """
    Read a wav file using scipy.io.wavfile
    """
    if wavfile.endswith('.wav'):
        rate, sig = wav.read(wavfile)
    elif wavfile.endswith('.flac'):
        sig, rate = soundfile.read(wavfile)
    else:
        raise IOError('NOT support file type or not a filename: {}'.format(wavfile))
    # data_name = os.path.splitext(os.path.basename(wavfile))[0]
    return rate, sig

# log-mel特征提取
def get_spectrograms(fpath, use_path=True):
    sr = 16000  # 16000  # keda, thchs30, aishell
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.05  # seconds
    frame_length = 0.1  # seconds
    hop_length = int(sr * frame_shift)  # samples.
    win_length = int(sr * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''

    # Loading sound file
    if use_path:
        y, sr = librosa.load(fpath, sr=sr)
        # with open("x.bin", 'wb') as fp:
        #     for i in range(len(y)):
        #         print("y[", i, "]: ", y[i])
        #         bs = struct.pack("f", y[i])
        #         # a = struct.pack('B', i)
        #         fp.write(bs)
    else:
        y, sr = fpath, sr
    # print("y.shape: ", y.shape)
    # print("sr: ", sr)

    time1 = time.time()
    # Trimming
    # y, _ = librosa.effects.trim(y)

    # Preemphasis pre-emphasis，预加重
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stftz
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)
    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)
    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)

    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    #mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    #mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    #mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    #
    # mel = mel[:len(mel) // r * r].reshape([len(mel) // r, r * n_mels])
    #mag = mag[:len(mag) // r * r]  # .reshape([len(mag)//hp.r,hp.r*1025])

    # time2 = time.time()
    # print("cost time:", time2-time1)

    return mel#, mag


import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage

# from scipy.io import wavfile
# from scipy.signal import butter, lfilter
# from audio_util import *

# from python_speech_features import mfcc
# from python_speech_features import delta
# from python_speech_features import logfbank
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import pdb
# import wavio
### Parameters ###
import scipy.signal
import torch

def audio2vector(audio_filename, dim_feature):
    '''
    Turn an audio file into feature representation.
    16k wav, size 283K -> len 903
    '''
    # rate, sig = load_wavfile(audio_filename)

    sig, rate = librosa.load(audio_filename, sr=16000)

    # Get fbank coefficients. numcep is the feature size
    sig = sig.reshape(sig.shape[0], 1)
    orig_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature).astype(np.float32)
    orig_inputs = (orig_inputs - np.mean(orig_inputs)) / np.std(orig_inputs)

    return orig_inputs


WINDOWS = {'hamming': scipy.signal.hamming,
           'hann': scipy.signal.hann,
           'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

audio_data = sorted(glob(os.path.join("/home/wangye/wangye/data/LibriSpeech/train-clean-360")))

for root, dirs, files in os.walk("/home/wangye/wangye/data/LibriSpeech/train-clean-360"):

    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list

    # 遍历所有的文件夹
    for dir in dirs:
        nowdr = root +'/'+ dir
        for root2, dirs2, _ in os.walk(nowdr):
            for dir2 in dirs2:
                nowdr2 = root2 + '/' + dir2
                for root3, dirs3, file in os.walk(nowdr2):
                    for fl in file:
                        fname = root3 + '/' + fl
                        if fname[-4:] == 'flac':
                            #   logspec = get_spectrograms(fname, True)
                            fbank = audio2vector(fname, 80)

                            out_file = fname[:-4] + "npy"  # 37 33
                            with open(out_file, 'wb') as f:
                                np.save(f, fbank)
                            # print(f)
                            print(len(fbank))


