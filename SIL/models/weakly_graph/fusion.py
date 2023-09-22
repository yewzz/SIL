import torch
import torch.nn.functional as F
from torch import nn

from models.modules import DynamicGRU, CrossGate, TanhAttention
from models.weakly_graph.prop import SparsePropMaxPool


class BaseFusion(nn.Module):
    def __init__(self, config):
        super(BaseFusion, self).__init__()
        hidden_size = config['hidden_size']
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)
        self.prop = SparsePropMaxPool(config['SparsePropMaxPool'])

    def forward(self, textual_input, text_len, textual_mask,
                visual_input, visual_len, visual_mask, fast_weights=None,
                **kwargs):
        map_h, map_mask = self.prop(visual_input.transpose(1, 2))
        map_h = self.vis_conv(map_h)
        txt_h = torch.stack([textual_input[i][l - 1] for i, l in enumerate(text_len)])
        txt_h = txt_h[:, :, None, None]
        fused_h = F.normalize(txt_h * map_h) * map_mask
        return fused_h, map_mask

    def reset_parameters(self):
        self.vis_conv.reset_parameters()


class BetterFusion(nn.Module):
    def __init__(self, config):
        super(BetterFusion, self).__init__()
        hidden_size = config['hidden_size']
        self.fuse_attn = TanhAttention(hidden_size)
        self.fuse_gate = CrossGate(hidden_size)
        self.fuse_gru = DynamicGRU(hidden_size * 2, hidden_size // 2,
                                   num_layers=1, bidirectional=True, batch_first=True)
        self.fuse_fc = nn.Linear(2 * hidden_size, hidden_size)


    def reset_parameters(self):
        self.fuse_attn.reset_parameters()
        self.fuse_gate.reset_parameters()
        self.fuse_gru.reset_parameters()

    def forward(self, textual_input, text_len, textual_mask, visual_input,
                visual_len=None, visual_mask=None, fast_weights=None, attn_weight=None,
                **kwargs):

        if False and fast_weights is not None:

            return None, None
        else:

            if attn_weight is not None:
                agg_txt_h, attn_weight, agg_2_h = self.fuse_attn(visual_input, textual_input, textual_mask, attn_weight=attn_weight)  # [b,64,512], [b,64,20], [b,20,512]
            else:
                agg_txt_h, attn_weight, agg_2_h = self.fuse_attn(visual_input, textual_input, textual_mask)  # [b,64,512], [b,64,20], [b,20,512]


            visual_h, agg_txt_h = self.fuse_gate(visual_input, agg_txt_h)
            #agg_txt_h, attn_weight, agg_2_h = self.fuse_attn(visual_input, textual_input, textual_mask,
            #                                                 attn_weight=attn_weight)
            #visual_input = self.feat_encoder(visual_input, visual_input)
            x = torch.cat([visual_h, agg_txt_h], -1)
            x = self.fuse_gru(x, None)
            fused_h = x

            return fused_h, attn_weight, textual_input, visual_input

