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
from python_speech_features import mfcc, logfbank
import soundfile

def load_wavfile(wavfile):
    """
    Read a wav file using scipy.io.wavfile
    """
    if wavfile.endswith('.wav'):
        rate, sig = wav.read(wavfile)
    elif wavfile.endswith('.flac'):
        sig, rate = soundfile.read(wavfile, dtype='int16')
    else:
        raise IOError('NOT support file type or not a filename: {}'.format(wavfile))
    # data_name = os.path.splitext(os.path.basename(wavfile))[0]
    return rate, sig

def audio2vector(audio_filename, dim_feature):
    '''
    Turn an audio file into feature representation.
    16k wav, size 283K -> len 903
    '''
    rate, sig = load_wavfile(audio_filename)

    # Get fbank coefficients. numcep is the feature size
    orig_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature).astype(np.float32)
    orig_inputs = (orig_inputs - np.mean(orig_inputs)) / np.std(orig_inputs)

    return orig_inputs

WINDOWS = {'hamming': scipy.signal.hamming,
           'hann': scipy.signal.hann,
           'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

audio_data = sorted(glob(os.path.join("/home/wangye/wangye2/wangye/wavs_16k_mixall/", "*.wav")))
print(len(audio_data))
# Grab your wav and filter it
maxlen = 0
for ad in audio_data:#[4000:5000]:


    # # Only use a short clip for our demo
    # # pdb.set_trace()
    # try:
    #     # pdb.set_trace()
    #     wave_obj = wavio.read(ad)
    #     rate = wave_obj.rate
    #     sig = np.squeeze(wave_obj.data)
    #     # (rate,sig) = wav.read(ad)
    # except TypeError:
    #     # print(ad)
    #     (rate,sig) = wav.read(ad)
    # # only short than 10 seconds
    # if np.shape(sig)[0]/float(rate) > 10:
    #     sig = sig[0:rate*10]
    # # Mel-filter bank
    # sig = sig - np.mean(sig)
    # fbank_feat = logfbank(sig, rate, winlen=0.025,\
    #     winstep=0.01,nfilt=40,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)
    #
    # # if fbank_feat.shape[0] < 1024:
    # #     # pdb.set_trace()
    # #     zero_pad = np.zeros((1024-fbank_feat.shape[0], 40))
    # #     fbank_feat = np.concatenate([fbank_feat, zero_pad], 0)
    # #
    # # maxlen = max(maxlen, fbank_feat.shape[0])
    fbank = audio2vector(ad, 80)
    x = ad[37:-4]
    #out_file = "/home1/yeshangwei/wangye/data/Audio/mel_np_2022/" + ad[33:-4] + ".npy"  # 37 33
    out_file = "/home/wangye/wangye2/wangye/TIP2021-erase/mel_np_2022_mixall/" + ad[44:-4] + ".npy"
    with open(out_file, 'wb') as f:
        np.save(f, fbank)
    print(f)

# ./data/flickr_audio/wavs/2865703567_52de2444f2_0.wav
print(maxlen)