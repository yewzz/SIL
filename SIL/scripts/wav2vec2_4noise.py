import soundfile as sf
import torch
# !pip install transformers
# !pip install dataset
import fairseq
import librosa
import glob
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from datasets import load_dataset

import random
# load model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

# model = Wav2Vec2Model.from_pretrained("/home/wangye/wangye2/wangye/wav2vec2_vox_960h_new.pt")


# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


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


new_path = '/home/wangye/wangye2/wangye/wavs_16k_noise/*'
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == '__main__':
    import soundfile as sf
    import torch
    from datasets import load_dataset
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecModel
    import fairseq

    from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model

    # from fairseq.dataclass.utils import convert_namespace_to_omegaconf

    fname = '/home/wangye/wangye2/wangye/wav2vec_big_960h.pt'
    cp = torch.load(fname)
    model = Wav2Vec2Model.build_model(cp['args'], None).cuda(3)
    model.load_state_dict(cp['model'], strict=False)
    model.eval()

    noise_files = get_noise_files(get_all_noise_files())
    cnt = 0
    for f in sorted(glob.iglob(new_path)):
        # wav_input_16khz = torch.randn(1, 10000)

        cnt = cnt + 1
        # if cnt < 60000:
        #     continue
        # if cnt == 60000:
        #     break
        #wav_input_16khz, rate = librosa.load(f, sr=16000)
        nf = random.choice(noise_files)

        wav_input_16khz, fs = librosa.load(nf, sr=16000)
        #plt.plot(wav)
        maxlen = min(160000, wav_input_16khz.shape[-1])
        wav_input_16khz = wav_input_16khz.reshape(1, -1)[0, :maxlen]

        # y, _ = librosa.effects.trim(wav_input_16khz, top_db=5)
        # wav_fn = f"{processed_dir}/wav_inputs/{basename}.wav"
        # y = y.write("./p225_004_2_L_2.wav", y, 16000)
        # sf.write('stereo_file1.wav', y, samplerate=16000)
        # print(librosa.get_duration(y), librosa.get_duration(wav_input_16khz))
        wav_input_16khz = torch.from_numpy(wav_input_16khz).view(1, -1).cuda(3)
        # retrieve logits
        c = model.forward(wav_input_16khz, mask=False, features_only=True)['x']

        out_file = "/home/wangye/wangye2/wangye/TIP2021-erase/wav2vec2.0_onlynoise_np/" + f[37:-4] + ".npy"
        out_file_path = Path(out_file)
        if out_file_path.exists():
            continue

        np_arr = c.detach().cpu().numpy()
        with open(out_file, 'wb') as f:
            np.save(f, np_arr)
        print(f)

