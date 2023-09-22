import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from models.modules.cq_attention import FeatureEncoder
from models.modules import DynamicGRU
from models.modules.transformer.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from models.modules.transformer import DualTransformer, TransformerEncoder

from models.weakly_graph.loss import info_nce

class AudioEncoder(nn.Module):
    def __init__(self, i_chans=512, idim=80, n_chans=512, out_dim=256, kernel_size=5):
        super(AudioEncoder, self).__init__()
        self.n_layers = 3
        self.convs = nn.ModuleList(nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5),
            nn.Conv1d(512, 512, kernel_size=8, stride=4),
            nn.Conv1d(512, 512, kernel_size=8, stride=4),
            nn.Conv1d(512, 512, kernel_size=4, stride=2),
            nn.Conv1d(512, 512, kernel_size=4, stride=2),
            nn.Conv1d(512, 512, kernel_size=4, stride=2),
        ))
        self.conv_relu = nn.GELU()
        self.conv_layernorm = nn.ModuleList([nn.LayerNorm(512) for _ in range(6)])

        self.new_enc = Conv2dSubampling(1, 300)#.requires_grad_(False)
        self.dim = 300
        self.input_dim = 80
        self.encoder_dim = self.dim
        self.ae_encoder = TransformerEncoder(4, self.dim, 6, 0.1, pre_ln=True)#.requires_grad_(False)
        self.ae_encoder2 = TransformerEncoder(2, self.dim, 6, 0.1, pre_ln=True)
        self.project = nn.Linear(self.encoder_dim * (((self.input_dim - 1) // 2 - 1) // 2), self.dim)#.requires_grad_(False)
        self.project2 = nn.Linear(512, 300)
        self.pos_encoder = SinusoidalPositionalEmbedding(self.dim, 0, 500)#.requires_grad_(False)

    def _mask_audio(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        # token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = l // 2
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            #p = weights[i, :l].cpu().numpy()
            # print(p)
            choices = np.random.choice(np.arange(0, l ), num_masked_words, replace=False)
            masked_words[-1][choices] = 1
        # exit(0)
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words

    def forward(self, audio_input, audio_input_wv, audio_len=None, audio_len_wv=None, audio_mask=None, init_memory=None, **kwargs):

        # audio_input [bsz, 1024, 40]
        # audio_input_wv = F.normalize(audio_input_wv, dim=-1) # normalize for wave
        # standarization
        means_wv = audio_input_wv.mean(dim=-1, keepdim=True)
        stds_wv = audio_input_wv.std(dim=-1, keepdim=True)
        audio_input_wv = (audio_input_wv - means_wv) / (stds_wv + 1e-10)

        audio_feat_wv = audio_input_wv.unsqueeze(1)
        for idx, conv in enumerate(self.convs):
            audio_feat_wv = conv(audio_feat_wv) # [64, 300, 116]
            audio_feat_wv = audio_feat_wv.transpose(-2, -1)
            audio_feat_wv = self.conv_layernorm[idx](audio_feat_wv)
            audio_feat_wv = audio_feat_wv.transpose(-2, -1)
            audio_feat_wv = self.conv_relu(audio_feat_wv)
        audio_feat_wv = audio_feat_wv.transpose(1, 2)

        ori_audio_len = audio_len
        ori_audio_mask = generate_mask(audio_input, audio_len)
        audio_feat = self.new_enc(audio_input, audio_len) # [bsz, 64, 512] [bsz, 256, 128]
        #audio_feat = audio_feat#, audio_feat_mid.transpose(1, 2), audio_feat_mem.transpose(1, 2)  # [bsz, 64, 512] [bsz, 256, 128]
        audio_len = audio_len // 4 - 1
        audio_len[audio_len <= 0] = 1

        audio_len_wv = audio_len_wv // 640 - 2
        audio_len_wv[audio_len_wv <= 0] = 1
        audio_mask = generate_mask(audio_feat, audio_len).unsqueeze(1).expand(-1, 127, -1)
        audio_mask_wv = generate_mask(audio_feat_wv, audio_len_wv).unsqueeze(1).expand(-1, 91, -1)
        # audio_chunk_mask = audio_mask.unsqueeze(-2) * self.chunk_mask

        audio_project = self.project(audio_feat)
        audio_pos = self.pos_encoder(audio_project)
        audio_project = audio_project + audio_pos

        audio_project_wv = self.project2(audio_feat_wv)
        audio_pos_wv = self.pos_encoder(audio_project_wv)
        audio_project_wv = audio_project_wv + audio_pos_wv

        audio_encoded, _ = self.ae_encoder(audio_project, audio_mask)
        audio_encoded_wv, _ = self.ae_encoder2(audio_project_wv, audio_mask_wv)

        # if audio_encoded.dim() == 3:
        #     audio_sent_encoded = audio_encoded.transpose(1, 2).unsqueeze(1) # [64, 1, 768, 128]
        # audio_sent_encoded = self.sent_conv(audio_sent_encoded) # [64, 768, 1, 128]
        # audio_sent_encoded = self.bn1(audio_sent_encoded)
        # audio_sent_encoded = audio_sent_encoded.squeeze(2).transpose(1, 2)

        # audio_sent_encoded = self.gru(audio_encoded, audio_len)
        # audio_feat = self.gru(audio_feat, audio_len)

        return audio_encoded, audio_encoded_wv


class Conv2dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self, in_channels: int, out_channels: int, layer=2) -> None:
        super(Conv2dSubampling, self).__init__()
        if layer == 2:
            self.convs = nn.ModuleList(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            ))
            self.conv_layernorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(2)])
            self.conv_relu = nn.GELU()
        else:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

    def forward(self, inputs, input_lengths=None):
        outputs = inputs.unsqueeze(1)
        for idx, conv in enumerate(self.convs):
            outputs = conv(outputs) # [bsz, 1, len, dim]
            outputs = outputs.permute(0, 2, 3, 1).contiguous()
            outputs = self.conv_layernorm[idx](outputs)
            outputs = outputs.permute(0, 3, 1, 2).contiguous()
            outputs = self.conv_relu(outputs)
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        # output_lengths = input_lengths >> 2
        # output_lengths -= 1

        return outputs#, output_lengths


def generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask