import torch.nn as nn
import torch


# class cif_function(nn.Module):
#     def __init__(self,  input, beta = 1.0, tail_thres = 0.5, padding_mask = None, target_lengths = None, eps = 1e-4):
#
#     r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
#     https://arxiv.org/abs/1905.11235
#     Args:
#         input (Tensor): (N, S, C) Input features to be integrated.
#         alpha (Tensor): (N, S) Weights corresponding to each elements in the
#             input. It is expected to be after sigmoid function.
#         beta (float): the threshold used for determine firing.
#         tail_thres (float): the threshold for determine firing for tail handling.
#         padding_mask (Tensor, optional): (N, S) A binary mask representing
#             padded elements in the input.
#         target_lengths (Tensor, optional): (N,) Desired length of the targets
#             for each sample in the minibatch.
#         eps (float, optional): Epsilon to prevent underflow for divisions.
#             Default: 1e-4
#     Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
#         cif_out (Tensor): (N, T, C) The output integrated from the source.
#         cif_lengths (Tensor): (N,) The output length for each element in batch.
#         alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
#             Can be used to compute the quantity loss.
#         delays (Tensor): (N, T) The expected delay (in terms of source tokens) for
#             each target tokens in the batch.
#         tail_weights (Tensor, optional): (N,) During inference, return the tail.
#     """
#         self.
#     B, S, C = input.size()
#     assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
#     prob_check(alpha)
#
#     dtype = alpha.dtype
#     alpha = alpha.float()
#     if padding_mask is not None:
#         padding_mask = padding_mask.bool()
#         alpha = alpha.masked_fill(padding_mask, 0)
#
#     if target_lengths is not None:
#         feat_lengths = target_lengths.long()
#         desired_sum = beta * target_lengths.type_as(input) + eps
#         alpha_sum = alpha.sum(1)
#         alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
#         T = feat_lengths.max()
#     else:
#         alpha_sum = alpha.sum(1)
#         feat_lengths = (alpha_sum / beta).floor().long()
#         T = feat_lengths.max()
#
#     # aggregate and integrate
#     csum = alpha.cumsum(-1)
#     with torch.no_grad():
#         # indices used for scattering
#         right_idx = (csum / beta).floor().long().clip(max=T)
#         left_idx = right_idx.roll(1, dims=1)
#         left_idx[:, 0] = 0
#
#         # count # of fires from each source
#         fire_num = right_idx - left_idx
#         extra_weights = (fire_num - 1).clip(min=0)
#
#     # The extra entry in last dim is for
#     output = input.new_zeros((B, T + 1, C))
#     delay = input.new_zeros((B, T + 1))
#     source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
#     zero = alpha.new_zeros((1,))
#
#     # right scatter
#     fire_mask = fire_num > 0
#     right_weight = torch.where(
#         fire_mask,
#         csum - right_idx.type_as(alpha) * beta,
#         zero
#     ).type_as(input)
#     # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
#     output.scatter_add_(
#         1,
#         right_idx.unsqueeze(-1).expand(-1, -1, C),
#         right_weight.unsqueeze(-1) * input
#     )
#     delay.scatter_add_(
#         1,
#         right_idx,
#         right_weight * source_range / beta
#     )
#
#     # left scatter
#     left_weight = (
#         alpha - right_weight - extra_weights.type_as(alpha) * beta
#     ).type_as(input)
#     output.scatter_add_(
#         1,
#         left_idx.unsqueeze(-1).expand(-1, -1, C),
#         left_weight.unsqueeze(-1) * input
#     )
#     delay.scatter_add_(
#         1,
#         left_idx,
#         left_weight * source_range / beta
#     )
#
#     # extra scatters
#     if extra_weights.ge(0).any():
#         extra_steps = extra_weights.max().item()
#         tgt_idx = left_idx
#         src_feats = input * beta
#         for _ in range(extra_steps):
#             tgt_idx = (tgt_idx + 1).clip(max=T)
#             # (B, S, 1)
#             src_mask = (extra_weights > 0)
#             output.scatter_add_(
#                 1,
#                 tgt_idx.unsqueeze(-1).expand(-1, -1, C),
#                 src_feats * src_mask.unsqueeze(2)
#             )
#             delay.scatter_add_(
#                 1,
#                 tgt_idx,
#                 source_range * src_mask
#             )
#             extra_weights -= 1
#
#     # tail handling
#     if target_lengths is not None:
#         # training time -> ignore tail
#         output = output[:, :T, :]
#         delay = delay[:, :T]
#     else:
#         # find out contribution to output tail
#         # note: w/o scaling, extra weight is all 0
#         zero = right_weight.new_zeros((1,))
#         r_mask = right_idx == feat_lengths.unsqueeze(1)
#         tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
#         l_mask = left_idx == feat_lengths.unsqueeze(1)
#         tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)
#
#         # a size (B,) mask that extends position that passed threshold.
#         extend_mask = tail_weights >= tail_thres
#
#         # extend 1 fire and upscale the weights
#         if extend_mask.any():
#             # (B, T, C), may have infs so need the mask
#             upscale = (
#                 torch.ones_like(output)
#                 .scatter(
#                     1,
#                     feat_lengths.view(B, 1, 1).expand(-1, -1, C),
#                     beta / tail_weights.masked_fill(~extend_mask, beta).view(B, 1, 1).expand(-1, -1, C),
#                 )
#                 .detach()
#             )
#             output *= upscale
#             feat_lengths += extend_mask.long()
#             T = feat_lengths.max()
#         output = output[:, :T, :]
#         delay = delay[:, :T]
#
#         # a size (B, T) mask to erase weights
#         tail_mask = torch.arange(T, device=output.device).unsqueeze(0) >= feat_lengths.unsqueeze(1)
#         output[tail_mask] = 0
#
#     return {
#         "cif_out": [output],
#         "cif_lengths": [feat_lengths],
#         "alpha_sum": [alpha_sum.to(dtype)],
#         "delays": [delay],
#         "tail_weights": [tail_weights] if target_lengths is None else []
#     }

class CifMiddleware(nn.Module):
    def __init__(self, threshold, embedding_dim, encoder_embed_dim, produce_weight_type='dense', conv_cif_width=5, apply_scaling=True, apply_tail_handling=True, tail_handling_firing_threshold=0.5):
        super().__init__()

        # Load configurations
        self.cif_threshold = threshold
        self.cif_output_dim = embedding_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.produce_weight_type = produce_weight_type
        self.conv_cif_width = conv_cif_width
        self.conv_cif_dropout = 0.1
        self.apply_scaling = apply_scaling
        self.apply_tail_handling = apply_tail_handling
        self.tail_handling_firing_threshold = tail_handling_firing_threshold

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                self.encoder_embed_dim,
                self.conv_cif_width,
                stride=1, padding=int(self.conv_cif_width / 2),
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.dense_proj = Linear(
                self.encoder_embed_dim, self.encoder_embed_dim).cuda()
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                self.encoder_embed_dim,
                self.conv_cif_width,
                stride=1, padding=int(self.conv_cif_width / 2),
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(
                p=self.conv_cif_dropout).cuda()
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()
        else:
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer (if encoder_embed_dim is not equal to cif_output_dim)
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False).cuda()

    def forward(self, encoder_raw_outputs, encoder_padding_mask, target_lengths):
        """
        Args:
            encoder_outputs: a dictionary that includes
                encoder_raw_out:
                    the raw outputs of acoustic encoder, with shape B x T x C
                encoder_padding_mask:
                    the padding mask (whose padded regions are filled with ones) of encoder outputs, with shape B x T
            target_lengths: the length of targets (necessary when training), with shape B
        Return:
            A dictionary:
                cif_out:
                    the cif outputs
                cif_out_padding_mask:
                    the padding infomation for cif outputs (whose padded regions are filled with zeros)
                quantity_out:
                    the sum of weights for the calculation of quantity loss
        """

        # Collect inputs
        # encoder_raw_outputs = encoder_outputs["encoder_raw_out"]        # B x T x C
        # encoder_padding_mask = encoder_outputs["encoder_padding_mask"]  # B x T

        # Produce weights for integration (accumulation)
        if self.produce_weight_type == "dense":
            conv_out = self.conv(encoder_raw_outputs.permute(0, 2, 1)).permute(0, 2, 1)
            proj_out = self.dense_proj(conv_out)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)
        # weight has shape B x T x 1

        not_padding_mask = 1 - encoder_padding_mask
        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()  # weight has shape B x T
        org_weight = weight
        ori_weight_np = org_weight.detach().cpu().numpy()
        # Apply scaling strategies
        if self.training and self.apply_scaling and target_lengths is not None:
            # Conduct scaling when training
            weight_sum = weight.sum(-1)             # weight_sum has shape B
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1)    # normalize_scalar has shape B x 1
            weight = weight * normalize_scalar

        # Prepare for Integrate and fire
        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        accumulated_weights = torch.zeros(batch_size, 0).cuda()
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = torch.zeros([batch_size]).cuda() if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = \
                torch.zeros([batch_size, encoder_embed_dim]).cuda() if i == 0 else accumulated_states[:, i - 1, :]

            # Decide whether to fire a boundary
            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(dim=-1)
            # cur_is_fired with shape B x 1

            # Update the accumulated weights
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape B x 1
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            # prev_accumulated_weight also has shape B x 1
            remained_weight = torch.ones_like(prev_accumulated_weight).cuda() - prev_accumulated_weight
            # remained_weight with shape B x 1

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight)  # B x 1

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :])  # B x C

            # Obtain fired state of current step:
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).cuda())  # B x C

            # Handle the tail
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than tail_handling_firing_threshold,
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i == padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                    # shape B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim]) <= self.tail_handling_firing_threshold,
                        # shape B x C
                        torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                        # less equal than tail_handling_firing_threshold, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than tail_handling_firing_threshold, normalized and kept.
                    ), cur_fired_state)
                # shape B x T

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero vectors
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).cuda() >
                padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).cuda(), cur_fired_state)

            # Update accumulation-related values: T_c stands for the length of integrated features
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1)                       # B x T_c
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1)     # B x T_c x C
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1)                 # B x T_c x C

        # Extract cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()    # B x T_c
        fired_utt_length = fired_marks.sum(-1)                          # B
        fired_max_length = fired_utt_length.max().int()                 # The maximum of fired times in current batch
        cif_outputs = torch.zeros([0, fired_max_length, encoder_embed_dim]).cuda()

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
            assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == index] for index in range(num_partitions)]

        # Loop over all samples
        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]             # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)         # The total number of firing
            pad_length = fired_max_length - cur_utt_length  # Get padded length
            cur_utt_output = torch.cat(
                (cur_utt_output, torch.full([pad_length, encoder_embed_dim], 0.0).cuda()), dim=0
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to 1 x T_c x C

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # cif_out_padding_mask has shape B x T_c, where locations with value 0 are the padded locations.

        if self.training:
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        return cif_outputs, cif_out_padding_mask, quantity_out
        # "cif_out": cif_outputs,                         # B x T_c x C
        # "cif_out_padding_mask": cif_out_padding_mask,   # B x T_c
        # "quantity_out": quantity_out                    # B



def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


import torch
from typing import Optional, Tuple
from torch import Tensor


def prob_check(tensor, eps=1e-10, neg_inf=-1e8, logp=False):
    assert not torch.isnan(tensor).any(), (
        "Nan in a probability tensor."
    )
    # Add the eps here to prevent errors introduced by precision
    if logp:
        assert tensor.le(0).all() and tensor.ge(neg_inf).all(), (
            "Incorrect values in a log-probability tensor"
            ", -inf <= tensor <= 0"
        )
    else:
        assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
            "Incorrect values in a probability tensor"
            ", 0.0 <= tensor <= 1.0"
        )


def cif_function(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235
    Args:
        input (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            input. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the input.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4
    Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
        cif_out (Tensor): (N, T, C) The output integrated from the source.
        cif_lengths (Tensor): (N,) The output length for each element in batch.
        alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays (Tensor): (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights (Tensor, optional): (N,) During inference, return the tail.
    """

    B, S, C = input.size()
    assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
    prob_check(alpha)

    dtype = alpha.dtype
    alpha = alpha.float()
    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        alpha = alpha.masked_fill(padding_mask, 0)

    if target_lengths is not None:
        feat_lengths = target_lengths.long()
        desired_sum = beta * target_lengths.type_as(input) + eps
        alpha_sum = alpha.sum(1)
        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        T = feat_lengths.max()
    else:
        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

    # aggregate and integrate
    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # indices used for scattering
        right_idx = (csum / beta).floor().long().clip(max=T)
        left_idx = right_idx.roll(1, dims=1)
        left_idx[:, 0] = 0

        # count # of fires from each source
        fire_num = right_idx - left_idx
        extra_weights = (fire_num - 1).clip(min=0)

    # The extra entry in last dim is for
    output = input.new_zeros((B, T + 1, C))
    delay = input.new_zeros((B, T + 1))
    source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = fire_num > 0
    right_weight = torch.where(
        fire_mask,
        csum - right_idx.type_as(alpha) * beta,
        zero
    ).type_as(input)
    # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
    output.scatter_add_(
        1,
        right_idx.unsqueeze(-1).expand(-1, -1, C),
        right_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(
        1,
        right_idx,
        right_weight * source_range / beta
    )

    # left scatter
    left_weight = (
        alpha - right_weight - extra_weights.type_as(alpha) * beta
    ).type_as(input)
    output.scatter_add_(
        1,
        left_idx.unsqueeze(-1).expand(-1, -1, C),
        left_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(
        1,
        left_idx,
        left_weight * source_range / beta
    )

    # extra scatters
    if extra_weights.ge(0).any():
        extra_steps = extra_weights.max().item()
        tgt_idx = left_idx
        src_feats = input * beta
        for _ in range(extra_steps):
            tgt_idx = (tgt_idx + 1).clip(max=T)
            # (B, S, 1)
            src_mask = (extra_weights > 0)
            output.scatter_add_(
                1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                src_feats * src_mask.unsqueeze(2)
            )
            delay.scatter_add_(
                1,
                tgt_idx,
                source_range * src_mask
            )
            extra_weights -= 1

    # tail handling
    if target_lengths is not None:
        # training time -> ignore tail
        output = output[:, :T, :]
        delay = delay[:, :T]
    else:
        # find out contribution to output tail
        # note: w/o scaling, extra weight is all 0
        zero = right_weight.new_zeros((1,))
        r_mask = right_idx == feat_lengths.unsqueeze(1)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        l_mask = left_idx == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        # a size (B,) mask that extends position that passed threshold.
        extend_mask = tail_weights >= tail_thres

        # extend 1 fire and upscale the weights
        if extend_mask.any():
            # (B, T, C), may have infs so need the mask
            upscale = (
                torch.ones_like(output)
                .scatter(
                    1,
                    feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                    beta / tail_weights.masked_fill(~extend_mask, beta).view(B, 1, 1).expand(-1, -1, C),
                )
                .detach()
            )
            output *= upscale
            feat_lengths += extend_mask.long()
            T = feat_lengths.max()
        output = output[:, :T, :]
        delay = delay[:, :T]

        # a size (B, T) mask to erase weights
        tail_mask = torch.arange(T, device=output.device).unsqueeze(0) >= feat_lengths.unsqueeze(1)
        output[tail_mask] = 0

    return {
        "cif_out": output,
        "cif_lengths": feat_lengths,
        "alpha_sum": alpha_sum.to(dtype),
        "delays": delay,
        "tail_weights": tail_weights if target_lengths is None else None
    }


if __name__ == '__main__':
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


    cif_model = CifMiddleware(0.99, 256, 256).cuda()
    enc_out = torch.rand([4, 25, 256]).cuda()
    enc_len = torch.tensor([21, 25, 24, 16]).cuda()
    enc_mask = 1 - generate_mask(enc_out, enc_len)
    target_len = torch.tensor([12, 12, 8, 2]).cuda()

    cif_outputs, cif_out_padding_mask, quantity_out = cif_model(enc_out, enc_mask, target_len)
