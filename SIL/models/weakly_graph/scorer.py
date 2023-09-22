import torch
import torch.nn.functional as F
from torch import nn
# from torch_geometric.nn import GATConv, GCNConv
# from torch_geometric.utils import dense_to_sparse
from models.modules import DynamicGRU
import numpy as np

def get_padded_mask_and_weight(*args):
    if len(args) == 2:
        mask, conv = args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(),
                                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
    elif len(args) == 5:
        mask, k, s, p, d = args
        masked_weight = torch.round(
            F.conv2d(mask.clone().float(), torch.ones(1, 1, k, k).cuda(), stride=s, padding=p, dilation=d))
    else:
        raise NotImplementedError

    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]  # conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight

#
# class GraphConv(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.gcn = GCNConv(input_size, output_size)
#
#     def _get_buffer(self, x, graph, bsz, len_):
#         if not hasattr(self, 'buffer_edge_index'):
#             adj_mat = graph.new_zeros(x.size(0), x.size(0))
#             for i in range(bsz):
#                 adj_mat[i * len_:(i + 1) * len_, i * len_:(i + 1) * len_] = graph
#             edge_index, edge_attr = dense_to_sparse(adj_mat)
#             # print(edge_index.size(1) / bsz)
#             setattr(self, 'num_edges_per_graph', edge_index.size(1) // bsz)
#             setattr(self, 'buffer_edge_index', edge_index)
#         total_edges = getattr(self, 'num_edges_per_graph') * bsz
#         return getattr(self, 'buffer_edge_index')[:, :total_edges]
#
    # def forward(self, x, graph):
    #     bsz, len_, hid_dim = x.size()
    #     x = x.contiguous().view(-1, hid_dim)
    #     edge_index = self._get_buffer(x, graph, bsz, len_)
    #     res = x
    #     x = self.gcn(x, edge_index)
    #     x = F.relu(x) + res
#         return x.contiguous().view(bsz, len_, -1)
#
#
# # # modified
# class GATGraphConv(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         num_heads = 1
#         self.gat = GATConv(input_size, output_size // num_heads, heads=num_heads)
#
#     def _get_buffer(self, x, graph, bsz, len_):
#         if not hasattr(self, 'buffer_edge_index') or True:
#             adj_mat = graph.new_zeros(x.size(0), x.size(0))
#             for i in range(bsz):
#                 adj_mat[i * len_:(i + 1) * len_, i * len_:(i + 1) * len_] = graph
#             edge_index, edge_attr = dense_to_sparse(adj_mat)
#             assert edge_index.size(1) % bsz == 0
#             setattr(self, 'num_edges_per_graph', edge_index.size(1) // bsz)
#             setattr(self, 'buffer_edge_index', edge_index)
#         total_edges = getattr(self, 'num_edges_per_graph') * bsz
#         return getattr(self, 'buffer_edge_index')[:, :total_edges]
#
#     def forward(self, x, graph):
#         bsz, len_, hid_dim = x.size()
#         x = x.contiguous().view(-1, hid_dim) # [b*l, d]
#         edge_index = self._get_buffer(x, graph, bsz, len_) # [2, 256]
#         res = x # [b*l, d]
#         x = self.gat(x, edge_index) # [b*l, d] [32, 8]
#         #x = F.relu(x) + res
#         return x.contiguous().view(bsz, len_, -1)
#
#
# class GeneralGraphConv(nn.Module):
#     def __init__(self, gcn_class, **kwargs):
#         super().__init__()
#         self.gcn = gcn_class(**kwargs)
#
#     def _get_buffer(self, x, graph, bsz, len_):
#         if not hasattr(self, 'buffer_edge_index') or True:
#             adj_mat = graph.new_zeros(x.size(0), x.size(0))
#             for i in range(bsz):
#                 adj_mat[i * len_:(i + 1) * len_, i * len_:(i + 1) * len_] = graph
#             edge_index, edge_attr = dense_to_sparse(adj_mat)
#             assert edge_index.size(1) % bsz == 0
#             setattr(self, 'num_edges_per_graph', edge_index.size(1) // bsz)
#             setattr(self, 'buffer_edge_index', edge_index)
#         total_edges = getattr(self, 'num_edges_per_graph') * bsz
#         return getattr(self, 'buffer_edge_index')[:, :total_edges]
#
#     def forward(self, x, graph):
#         bsz, len_, hid_dim = x.size()
#         x = x.contiguous().view(-1, hid_dim)
#         edge_index = self._get_buffer(x, graph.squeeze(0), bsz, len_)
#         x = x.unsqueeze(1)
#         res = x
#         x = self.gcn(x, edge_index)
#         x = F.relu(x) + res
#         return x.contiguous().view(bsz, len_, -1)
#
#
# class MapGraph(nn.Module):
#     def __init__(self, config):
#         super(MapGraph, self).__init__()
#         input_size = config['input_size']
#         hidden_sizes = config['hidden_sizes']
#
#         self.convs = nn.ModuleList()
#
#         channel_sizes = [input_size] + hidden_sizes
#         for i, d in enumerate(hidden_sizes):
#             self.convs.append(GATGraphConv(channel_sizes[i], channel_sizes[i + 1]))
#
#         self.pred_layer = nn.Linear(channel_sizes[-1], 1)
#         # self.assess_layer = nn.Linear(channel_sizes[-1], 1)
#
#     def forward(self, props_h, props_graph, props, props_all, new_props_graph,**kwargs):
#         # add
#         # props_h = [batch_size, 153, d]
#         # from models.modules.graph_conv import get_affinity
#         # affinity = get_affinity(props_h) # [batch_size, 153, 153]
#         # props_graph = affinity # [batch_size, 153, 153]
#         #props_graph = 1 - torch.eye(props_h.size(1), props_h.size(1))
#         x = props_h
#         batch_size = x.size(0)
#         hidden_size = x.size(-1)
#         ori_num_clips = 64
#         #props_graph = props_graph.unsqueeze(0).expand(batch_size, -1, -1)
#         for c in self.convs:
#             x = c(x, props_graph)
#         # recover
#         #ori_map_h = x.new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)
#         #ori_map_h[:, :, props_all[:, 0], props_all[:, 1] - 1] = x.transpose(1, 2)
#         #x = ori_map_h[:, :, props[:, 0], props[:, 1] - 1].transpose(1, 2)
#
#         x_pred = self.pred_layer(x).squeeze(-1)
#         # x_assess = self.assess_layer(x).squeeze(-1)
#         return x_pred, torch.rand([x.size(0), 64]).cuda()# , x_assess
#
#     def reset_parameters(self):
#         self.pred_layer.reset_parameters()
#

class MapConv(nn.Module):
    def __init__(self, config):
        super(MapConv, self).__init__()
        input_size = config['input_size']
        hidden_sizes = config['hidden_sizes']
        kernel_sizes = config['kernel_sizes']
        strides = config['strides']
        paddings = config['paddings']
        dilations = config['dilations']
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size] + hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, p, d))

        self.first_layer = nn.Conv2d(hidden_sizes[-1], hidden_sizes[-1], 1, 1)
        self.pred_layer = nn.Conv2d(hidden_sizes[-1], 1, 1, 1)
        self.pred_layer2 = nn.Conv2d(hidden_sizes[-1], 1, 1, 1)
        self.fuse_gru = DynamicGRU(hidden_sizes[-1] * 2, hidden_sizes[-1] // 2,
                                   num_layers=1, bidirectional=True, batch_first=True)

        frame_props = []
        for i in range(64):
            frame_props.append(torch.tensor([i, i]))
        self.frame_props = torch.stack(frame_props, 0).cuda()

        self.num_clips = 64

        self.props = []

        #for i in range(64):
        #    self.props.append(np.asarray([i, i]).reshape(1, 2))
        tmp = [[1], [2], [2]]
        tmp[0].extend([1] * 15)
        tmp[1].extend([1] * 7)
        tmp[2].extend([1] * 7)
        # for i in range(0, 60):
        #     self.props.append(np.asarray([i, i + 4]).reshape(1, 2))
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
        self.props = torch.from_numpy(self.props).cuda()

        ####### construct grapn
        self.layers = nn.ModuleList()
        self.num_scale_layers = [16, 8, 8]
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
            low_idx = props_iou[i] < 0.5
            self.props_graph[i, low_idx] = 0
        self.props_graph = torch.from_numpy(self.props_graph).cuda()



    def forward(self, map_h, map_mask, props, test_props=None, **kwargs):
        if map_mask is not None:
            padded_mask = map_mask
            x = map_h

            for i, pred in enumerate(self.convs):
               x = torch.relu_(pred(x))
               padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
               x = x * masked_weight


            x_pred = self.pred_layer(x)

            if test_props is None:
                x_frame_pred = x_pred[:, 0, self.frame_props[:, 0], self.frame_props[:, 1]]
                x_pred = x_pred[:, 0, self.props[:, 0], self.props[:, 1] - 1] # [64, 1, 32, 32]

            else:
                new_list = []
                for i in range(x_pred.size(0)):
                    new_list.append(x_pred[i, 0, test_props[i, :, 0], test_props[i, :, 1] - 2])
                new_list = torch.stack(new_list, 0)
                x_pred = new_list

            return x_pred, x_frame_pred
        else:
            if map_h.size(2) == 512:
                fused_h = map_h.transpose(1, 2).unsqueeze(-1) # [batch_size, 512, 64, 1]
                x_pred = self.pred_layer2(fused_h).squeeze(-1).squeeze(1) # [batch_size, 512, 64, 1] -> [batch_size, 1, 64, 1]

            else:
                map_h = self.fuse_gru(map_h, seq_len=None)
                fused_h = map_h.transpose(1, 2).unsqueeze(-1)
                x_pred = self.pred_layer(fused_h).squeeze(-1).squeeze(1)
            return x_pred, None

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        self.pred_layer.reset_parameters()


class fusion_scorer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, fused_h):
        score = self.fc(fused_h) # [batch_size, 64, 512] -> [batch_size, 64, 1]
        return score.squeeze(-1)


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