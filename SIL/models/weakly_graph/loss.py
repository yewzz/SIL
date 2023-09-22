import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def sigmoid_and_normalize(scores):
    joint_prob = torch.sigmoid(scores)
    #joint_prob = scores

    min_prob, max_prob = joint_prob.min(dim=-1, keepdim=True)[0], \
                         joint_prob.max(dim=-1, keepdim=True)[0]
    joint_prob_norm = (joint_prob - min_prob + 1e-10) / (max_prob - min_prob + 1e-10)
    return joint_prob, joint_prob_norm


def bce_rescale_loss(scores, targets, min_iou=0.5, max_iou=1.0, bias=0.0, reduction='mean'):
    joint_prob, joint_prob_norm = sigmoid_and_normalize(scores)
    # joint_prob = joint_prob_norm

    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none')
    loss_value = loss.mean(dim=-1)
    if reduction == 'mean':
        loss_value = loss_value.mean(dim=-1)
    elif reduction == 'none':
        loss_value = loss_value
    else:
        loss_value = loss_value.sum(dim=-1)
    return loss_value, joint_prob_norm

def bce_scale_loss(scores, targets, min_iou=0.5, max_iou=1.0, bias=0.0, weight=None, reduction='mean'):
    joint_prob, joint_prob_norm = sigmoid_and_normalize(scores)
    #joint_prob = joint_prob_norm

    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob_mean = target_prob.mean(dim=-1)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0


    loss = F.binary_cross_entropy_with_logits(scores, target_prob, reduction='none')
    #loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') # KL loss better
    loss_value = loss.mean(dim=-1)
    if reduction == 'mean':
        loss_value = loss_value.mean(dim=-1)
    elif reduction == 'none':
        loss_value = loss_value
    else:
        loss_value = loss_value.sum(dim=-1)
    return loss_value, joint_prob_norm

def weakly_supervised_loss(pos_score,  clean_score, loss_rec, loss_dcor, map_iou, pos_weight, frame_feat, neg_score1, self_neg_score1, neg_score2, neg_weight2, neg_weight2_graph, weight_gt, map_weight, prob_mat, prob_mat_neg, start_prob, end_prob,  map_gt, erase_tri_loss, erase_tri_loss_verb, erase_tri_loss_noun, props,
                           num_cands, reg_gt,  log_fp=None, loss_meter=None, loss_kd=None, noisy=False, **kwargs):
    info = ''
    def calc_loss(score, rank_score=None, positive=True, confidence=False):
        bsz, num_clips = score.size()
        joint_prob, joint_prob_norm = sigmoid_and_normalize(score)

        # if rank_score is not None:
        #     joint_prob = joint_prob * rank_score
        joint_prob_np = joint_prob.detach().cpu().numpy()
        idx = torch.argsort(joint_prob_norm, dim=-1, descending=True)
        props1 = props[idx[:, :num_cands]].contiguous()  # [b, 200, 2]
        props2 = props1[:, 0]
        props2 = props2.unsqueeze(1).expand(bsz, num_cands, 2).contiguous().view(bsz * num_cands, 2)
        props1 = props1.view(bsz * num_cands, 2)
        iou = calculate_IoU_batch((props2[:, 0], props2[:, 1]), (props1[:, 0], props1[:, 1]))
        iou = iou.contiguous().view(bsz, num_cands)
        iou = iou.type_as(joint_prob_norm)

        sort_idx = torch.argsort(iou, dim=-1, descending=True)[:, :kwargs['topK']]
        idx1 = idx.gather(dim=-1, index=sort_idx) # [64, 20]
        idx1 = idx[:, :kwargs['topK']]
        if positive:
            tmp = joint_prob.gather(dim=-1, index=idx1) # [64, 153]
        else:
            tmp = joint_prob.gather(dim=-1, index=idx1)

        align_score = tmp.mean(dim=-1)

        if positive:
            tmp1 = joint_prob_norm.mean() # global loss
            sort_idx_top = torch.argsort(iou, dim=-1, descending=True)[:, :32]
            idx1_top = idx.gather(dim=-1, index=sort_idx_top)
            tmp_top = joint_prob_norm.gather(dim=-1, index=idx1_top)
            #tmp1 = tmp_top.mean()
            nonlocal info
            info += 'soc {}, '.format(float(tmp1))
            # tmp1 = F.relu(joint_prob_norm.mean(dim=-1) - 0.6).mean()
            # tmp2 = (tmp * F.log_softmax(tmp, -1)).mean()
            # tmp2 = -(tmp.softmax(dim=-1) * tmp.log_softmax(dim=-1)).sum(dim=-1).mean()
            tmp2 = -(tmp.log_softmax(dim=-1).max(dim=-1)[0]).mean() # gap loss
            # tmp2 = F.log_softmax(tmp, -1).mean()
            norm_loss =  tmp1
            norm_loss_new = tmp2
        else:
            norm_loss_new = None
            norm_loss = None


        return joint_prob, joint_prob_norm, align_score, norm_loss, idx1, norm_loss_new



    joint_prob, joint_prob_norm, pos_score, norm_loss1, idx, norm_loss_new = calc_loss(pos_score, None, positive=True, confidence=False)
    joint_prob_neg , joint_prob_norm_neg, neg_score1, neg_norm_loss1,_ , _= calc_loss(neg_score1, self_neg_score1, positive=False)

    info += 'pos {}, neg1 {}, '.format(float(pos_score.mean(dim=-1)), float(neg_score1.mean(dim=-1)))
    #inter_loss = F.relu(neg_score1 - pos_score + 1.0).mean(dim=-1)
    inter_loss = (-torch.log(pos_score + 1e-10) + -torch.log(1.0 - neg_score1 + 1e-10)).mean()

    loss_meter['inter_loss2'].update(F.relu(neg_score1 - pos_score + 0.2).mean(dim=-1).item())
    loss_meter['inter_loss'].update(inter_loss.item())
    loss_meter['norm_loss1'].update(norm_loss1.item() + norm_loss_new.item())

    if log_fp is not None:
        log_fp.write(info + '\n')
        log_fp.flush()

    final_loss =  1 * inter_loss + kwargs['norm1'] * (1 * norm_loss1 + 1 * norm_loss_new)
    return final_loss, None,  None #0.1 * rank_loss


def weakly_supervised_loss_new(pos_score, neg_score1, props,
                           num_cands, log_fp=None, loss_meter=None, map_gt=None, **kwargs):

    # supervised
    # sup_loss, _ = bce_rescale_loss(pos_score, map_gt, min_iou=0.3, max_iou=0.7)
    # loss_meter['sup_loss'].update(sup_loss.item())
    # return sup_loss, sup_loss


    # weakly-supervised
    info = ''
    def calc_loss(score, positive=True):
        bsz, num_clips = score.size()
        joint_prob, joint_prob_norm = sigmoid_and_normalize(score)
        # joint_prob = joint_prob_norm

        idx = torch.argsort(joint_prob_norm, dim=-1, descending=True)
        props1 = props[idx[:, :num_cands]].contiguous()  # [bsz, 200, 2]
        props2 = props1[:, 0]
        props2 = props2.unsqueeze(1).expand(bsz, num_cands, 2).contiguous().view(bsz * num_cands, 2)
        props1 = props1.view(bsz * num_cands, 2)
        iou = calculate_IoU_batch((props2[:, 0], props2[:, 1]), (props1[:, 0], props1[:, 1]))
        iou = iou.contiguous().view(bsz, num_cands)
        iou = iou.type_as(joint_prob_norm)

        sort_idx = torch.argsort(iou, dim=-1, descending=True)[:, :kwargs['topK']]
        idx1 = idx.gather(dim=-1, index=sort_idx)
        idx1 = idx[:, :kwargs['topK']]
        tmp = joint_prob.gather(dim=-1, index=idx1)
        # tmp = joint_prob.gather(dim=-1, index=idx)[:, :kwargs['topK']]
        align_score = tmp.mean(dim=-1)

        # log_fp.write('{}, {}\n'.format(positive, tmp[0]))
        if positive:
            tmp1 = joint_prob_norm.mean()
            nonlocal info
            info += 'soc {}, '.format(float(tmp1))
            # tmp1 = F.relu(joint_prob_norm.mean(dim=-1) - 0.6).mean()
            # tmp2 = (tmp * F.log_softmax(tmp, -1)).mean()
            tmp2 = -(tmp.softmax(dim=-1) * tmp.log_softmax(dim=-1)).sum(dim=-1).mean()
            tmp2 = -(tmp.log_softmax(dim=-1).max(dim=-1)[0]).mean()
            # tmp2 = F.log_softmax(tmp, -1).mean()
            # charades: 5e-2, 1e-1
            # norm_loss = 1e-2 * tmp1 + 1e-2 * tmp2
            norm_loss1 = tmp1 #+ 0.1 * tmp2
            norm_loss2 = 1.0 * tmp2
        else:
            norm_loss1 = None
            norm_loss2 = None

        return joint_prob_norm, align_score, norm_loss1, norm_loss2

    joint_prob_norm, pos_score, norm_loss1, norm_loss2 = calc_loss(pos_score, positive=True)

    _, neg_score1, _ , _= calc_loss(neg_score1, positive=False)

    inter_loss = (-torch.log(pos_score + 1e-10) + -torch.log(1.0 - neg_score1 + 1e-10)).mean()

    loss_meter['inter_loss2'].update(F.relu(neg_score1 - pos_score + 0.2).mean(dim=-1).item())
    loss_meter['inter_loss'].update(inter_loss.item())

    loss_meter['norm_loss1'].update(norm_loss1.item())
    loss_meter['norm_loss2'].update(norm_loss2.item())

    loss = inter_loss + 0.001 * (norm_loss1 + norm_loss2)# + 0.1 *  norm_loss2

    return loss, 0.01 * (norm_loss1 + norm_loss2)


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# contrast loss
class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1) # [bsz, dim]
        k = nn.functional.normalize(k, dim=1) # [bsz, dim]
        neg = neg.permute(0,2,1) # [bsz ,dim ,len]
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # [bsz, 1]
        l_neg = torch.einsum('nc,nck->nk', [q, neg]) # [bsz, 12]
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels) # -ln(1-x)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA']
        )

        loss = HA_refinement + HB_refinement
        return loss


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired',mask=None):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys) # [bsz, neg_num]

        elif negative_mode == 'paired':
            query = query.unsqueeze(1) # [bsz, 1, dim]
            negative_logits = query @ transpose(negative_keys) # [bsz, 1, dim] * [bsz, dim, neg_num]
            negative_logits = negative_logits.squeeze(1) # [bsz, neg_num]

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    #a = F.cross_entropy(logits / temperature, labels, reduction=reduction)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


