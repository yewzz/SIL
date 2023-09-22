import torch
from torch import nn

from models.modules import TanhAttention
from torch.functional import F
from models.modules import DynamicGRU
from models.modules.multihead_attention import MultiheadAttention


from models.modules.cross_gate import CrossGate
import numpy as np

def iou(pred, gt):
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * (inter + 1e-10) / (union + 1e-10)
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

class SparsePropMaxPool(nn.Module):
    def __init__(self, config, hidden_size=512):
        super(SparsePropMaxPool, self).__init__()
        self.num_scale_layers = config['num_scale_layers']

        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            if scale_idx == 0:
                first_layer = nn.MaxPool1d(1, 1)
            elif scale_idx == 1:
                first_layer = nn.MaxPool1d(3, 2)
            else:
                # first_layer = nn.MaxPool1d(5, 4)
                first_layer = nn.MaxPool1d(3, 2)
            rest_layers = [nn.MaxPool1d(2, 1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer] + rest_layers)
            self.layers.append(scale_layers)



        self.graph = torch.ones([32, 32]).cuda()
        self.fc = nn.Conv1d(512 * 3, 512, 1, 1)
        self.cross_gate = CrossGate(512)

        # graph
        acum_layers = 0
        stride = 1
        ori_num_clips = 64
        ori_map_mask = torch.zeros(ori_num_clips, ori_num_clips)
        for scale_idx, scale_layers in enumerate(self.layers):
            for i, layer in enumerate(scale_layers):
                stride = stride * layer.stride
                ori_s_idxs = list(range(0, ori_num_clips - acum_layers - i * stride, stride))
                ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]
                ori_map_mask[ori_s_idxs, ori_e_idxs] = 1
            acum_layers += stride * (len(scale_layers) + 1)
        self.props_all_pos = torch.nonzero(ori_map_mask[:, :])
        self.props_all_pos[:, 1] = self.props_all_pos[:, 1] + 1

        props_iou = iou(self.props_all_pos.tolist(), self.props_all_pos.tolist())  # shape:(153,153)这是2D-TAN的架势？
        self.props_graph = np.zeros_like(props_iou).astype(np.int32)
        sort_idx = np.argsort(-props_iou, -1)
        for i in range(self.props_all_pos.shape[0]):
            self.props_graph[i, sort_idx[i]] = 1
            low_idx = props_iou[i] < 0.6
            #self.props_graph[i, low_idx] = 0
        self.props_graph = torch.from_numpy(self.props_graph).cuda()

        self.num_clips = 64

        self.props = []
        tmp = [[1], [2], [2]]
        tmp[0].extend([1] * 15)
        tmp[1].extend([1] * 7)
        tmp[2].extend([1] * 7)
        acum_layers = 0
        stride = 1
        for scale_idx, strides in enumerate(tmp):
            for i, stride_i in enumerate(strides):
                stride = stride * stride_i
                keep = False

                if scale_idx == 0 and i in [7, 15]:
                    keep = True
                elif scale_idx == 1 and (i in [3, 7]):
                    keep = True
                elif scale_idx == 2 and (i in [3, 5, 7]):
                    keep = True

                if not keep:
                    continue
                ori_s_idxs = list(range(0, self.num_clips - acum_layers - i * stride, stride))
                ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]

                self.props.append(np.stack([ori_s_idxs, ori_e_idxs], -1))

            acum_layers += stride * (len(strides) + 1)
        self.props = np.concatenate(self.props, 0)  # shape:[153,2]
        self.props[:, 1] += 1


    def reset_parameters(self):
        pass

    def forward(self, x, props, props_graph, **kwargs):
        ori_x = x.transpose(1, 2) # [b, len, dim]
        batch_size, hidden_size, ori_num_clips = x.size()
        acum_layers = 0
        stride = 1
        ori_map_h = x.new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips) # [64, 512, 64, 64]
        ori_map_mask = x.new_zeros(batch_size, 1, ori_num_clips, ori_num_clips)
        for scale_idx, scale_layers in enumerate(self.layers):
            for i, layer in enumerate(scale_layers):
                stride = stride * layer.stride
                # print(x.shape)
                x = layer(x)
                ori_s_idxs = list(range(0, ori_num_clips - acum_layers - i * stride, stride))
                ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]
                ori_map_h[:, :, ori_s_idxs, ori_e_idxs] = x
                ori_map_mask[:, :, ori_s_idxs, ori_e_idxs] = 1
                # print(ori_s_idxs)
                # print(ori_e_idxs)
                # print('=====================')
            acum_layers += stride * (len(scale_layers) + 1)
        #ori_map_h = F.normalize(ori_map_h)

        #props_h = ori_map_h[:, :, self.props_all_pos[:, 0], self.props_all_pos[:, 1] - 1]
        props_h = ori_map_h[:, :, props[:, 0], props[:, 1] - 1]
        props_h = props_h.transpose(1, 2)

        ori_map_h[:, :, props[:, 0], props[:, 1] - 1] = props_h.transpose(1, 2)

        return props_h, ori_map_h, ori_map_mask

class SparsePropConv(nn.Module):
    def __init__(self, config):
        super(SparsePropConv, self).__init__()
        self.num_scale_layers = config['num_scale_layers']
        self.hidden_size = 512
        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            if scale_idx == 0:
                first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 1,1)
            elif scale_idx == 1:
                first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 3,2)
            else:
                # first_layer = nn.MaxPool1d(5, 4)
                first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 2,1)
            rest_layers = [nn.Conv1d(self.hidden_size, self.hidden_size, 2,1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer] + rest_layers)
            self.layers.append(scale_layers)

        self.cross_gate = CrossGate(512)

    def forward(self, x, props, textual_input, textual_mask, **kwargs):
        batch_size, hidden_size, ori_num_clips = x.size()

        acum_layers = 0
        stride = 1
        ori_map_h = x.new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)  # [64, 512, 64, 64]
        ori_map_mask = x.new_zeros(batch_size, 1, ori_num_clips, ori_num_clips)
        for scale_idx, scale_layers in enumerate(self.layers):
            for i, layer in enumerate(scale_layers):
                stride = stride * layer.stride[0]
                # print(x.shape)
                #x = layer(x)
                if i == 0:
                    x = x
                    ori_s_idxs = list(range(0, ori_num_clips - acum_layers - i * stride, stride))  # [0-31] [0-30]
                    ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]  # [0-31] [1-31]
                    ori_map_h[:, :, ori_s_idxs, ori_e_idxs] = x
                    ori_map_mask[:, :, ori_s_idxs, ori_e_idxs] = 1
                else:
                    index_1 = torch.tensor(list(range(0, ori_num_clips - acum_layers - i * stride, stride)))
                    index_2 = index_1 + 1
                    x_1, x_2 = x[:, :, index_1].contiguous().transpose(1, 2), x[:, :, index_2].contiguous().transpose(1, 2)
                    x_1, x_2 = self.cross_gate(x_1, x_2)
                    x = x_1 + x_2
                    x = x.contiguous().transpose(1, 2)
                    ori_s_idxs = list(range(0, ori_num_clips - acum_layers - i * stride, stride)) #[0-31] [0-30]
                    ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs] #[0-31] [1-31]
                    ori_map_h[:, :, ori_s_idxs, ori_e_idxs] = x
                    ori_map_mask[:, :, ori_s_idxs, ori_e_idxs] = 1
                # print(ori_s_idxs)
                # print(ori_e_idxs)
                # print('=====================')
            acum_layers += stride * (len(scale_layers) + 1)
        props_h = ori_map_h[:, :, props[:, 0], props[:, 1] - 1]
        props_h = props_h.transpose(1, 2)

        return props_h, ori_map_h, ori_map_mask


class SparsePropGate(nn.Module):
    def __init__(self, config):
        super(SparsePropGate, self).__init__()
        self.num_scale_layers = config['num_scale_layers']
        self.hidden_size = 512
        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            if scale_idx == 0:
                first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 1,1)
            elif scale_idx == 1:
                first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 3,2)
            else:
                # first_layer = nn.MaxPool1d(5, 4)
                first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 2,1)
            rest_layers = [nn.Conv1d(self.hidden_size, self.hidden_size, 2,1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer] + rest_layers)
            self.layers.append(scale_layers)

    def forward(self, x, props, textual_input, textual_mask, **kwargs):
        batch_size, hidden_size, ori_num_clips = x.size()

        acum_layers = 0
        stride = 1
        ori_map_h = x.new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)  # [64, 512, 64, 64]
        ori_map_mask = x.new_zeros(batch_size, 1, ori_num_clips, ori_num_clips)
        for scale_idx, scale_layers in enumerate(self.layers):
            for i, layer in enumerate(scale_layers):
                stride = stride * layer.stride[0]
                # print(x.shape)
                #index_0 = torch.linspace(0, ori_num_clips, steps=12).view((2, 2, 3))
                x = layer(x)
                ori_s_idxs = list(range(0, ori_num_clips - acum_layers - i * stride, stride))
                ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]
                ori_map_h[:, :, ori_s_idxs, ori_e_idxs] = x
                ori_map_mask[:, :, ori_s_idxs, ori_e_idxs] = 1
                # print(ori_s_idxs)
                # print(ori_e_idxs)
                # print('=====================')
            acum_layers += stride * (len(scale_layers) + 1)
        props_h = ori_map_h[:, :, props[:, 0], props[:, 1] - 1]
        props_h = props_h.transpose(1, 2)

        return props_h, ori_map_h, ori_map_mask



class DensePropMaxPool(nn.Module):
    def __init__(self, config):
        super(DensePropMaxPool, self).__init__()
        num_layers = config['num_layers']
        self.layers = nn.ModuleList(
            [nn.Identity()]
            + [nn.MaxPool1d(2, stride=1) for _ in range(num_layers - 1)]
        )
        self.num_layers = num_layers

    def forward(self, x, props, **kwargs):
        batch_size, hidden_size, num_clips = x.shape
        map_h = x.new_zeros(batch_size, hidden_size, num_clips, num_clips).cuda()
        map_mask = x.new_zeros(batch_size, 1, num_clips, num_clips).cuda()

        for dig_idx, pool in enumerate(self.layers):
            x = pool(x)
            start_idxs = [s_idx for s_idx in range(0, num_clips - dig_idx, 1)]
            end_idxs = [s_idx + dig_idx for s_idx in start_idxs]
            map_h[:, :, start_idxs, end_idxs] = x
            map_mask[:, :, start_idxs, end_idxs] = 1
        props_h = map_h[:, :, props[:, 0], props[:, 1] - 1]
        # map_mask[:, :, props[:, 0], props[:, 1] - 1] = 1
        # print(props[:, 0], props[:, 1] - 1)
        # exit(0)
        return props_h.transpose(1, 2), map_h, map_mask

