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

# load model and processor
#processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

#model = Wav2Vec2Model.from_pretrained("/home/wangye/wangye2/wangye/wav2vec2_vox_960h_new.pt")


# define function to read in sound file
def map_to_array(batch):
  speech, _ = sf.read(batch["file"])
  batch["speech"] = speech
  return batch


new_path = '/home/wangye/wangye2/wangye/wavs_16k_mixall/*'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    import soundfile as sf
    import torch
    from datasets import load_dataset
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecModel
    import fairseq

    from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
    #from fairseq.dataclass.utils import convert_namespace_to_omegaconf

    fname = '/home/wangye/wangye2/wangye/wav2vec_big_960h.pt'
    cp = torch.load(fname)
    model = Wav2Vec2Model.build_model(cp['args'], None).cuda(0)
    model.load_state_dict(cp['model'], strict=False)
    model.eval()

    cnt = 0
    for f in sorted(glob.iglob(new_path)):
        #wav_input_16khz = torch.randn(1, 10000)

        cnt = cnt + 1
        # if cnt < 60000:
        #     continue
        if cnt < 30000:
            continue
        wav_input_16khz, rate = librosa.load(f, sr=16000)

        # y, _ = librosa.effects.trim(wav_input_16khz, top_db=5)
        # wav_fn = f"{processed_dir}/wav_inputs/{basename}.wav"
        # y = y.write("./p225_004_2_L_2.wav", y, 16000)
        # sf.write('stereo_file1.wav', y, samplerate=16000)
        # print(librosa.get_duration(y), librosa.get_duration(wav_input_16khz))
        wav_input_16khz = torch.from_numpy(wav_input_16khz).view(1, -1).cuda(0)
        # retrieve logits
        c = model.forward(wav_input_16khz, mask=False, features_only=True)['x']

        out_file = "/home/wangye/wangye2/wangye/TIP2021-erase/wav2vec2.0_mixall_np/" + f[44:-4] + ".npy"
        out_file_path = Path(out_file)
        if out_file_path.exists():
            continue

        np_arr = c.detach().cpu().numpy()
        with open(out_file, 'wb') as f:
            np.save(f, np_arr)
        print(f)

