import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.weakly_graph.fusion as fusion
import models.weakly_graph.query_encoder as query_encoder
from models.weakly_graph.audio_encoder import AudioEncoder
import models.weakly_graph.scorer as scorer
import models.weakly_graph.video_encoder as video_encoder
from models.modules import TanhAttention
from models.weakly_graph.prop import SparsePropMaxPool,SparsePropConv
from models.modules.dynamic_rnn import DynamicGRU
from models.modules.cross_gate import CrossGate, SelfGate
from models.modules.transformer.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from scipy import ndimage
import math
from models.modules.multihead_attention import MultiheadAttention
from models.modules.transformer import TransformerEncoder


class WeaklyGraphNew(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Base Grounding Network
        self.video_encoder = getattr(video_encoder, config['VideoEncoder']['name'])(config['VideoEncoder'])
        self.query_encoder = getattr(query_encoder, config['QueryEncoder']['name'])(config['QueryEncoder'])

        self.audio_encoder = AudioEncoder().requires_grad_(True)
        self.fusion = getattr(fusion, config['Fusion']['name'])(config['Fusion'])
        self.fuse_attn = TanhAttention(512)
        self.fuse_gru = DynamicGRU(512 * 2, 512 // 2,
                                   num_layers=1, bidirectional=True, batch_first=True)

        self.prop = SparsePropMaxPool(config['Fusion']['SparsePropMaxPool'])
        self.scorer = getattr(scorer, config['Scorer']['name'])(config['Scorer'])

        # ASP module
        self.word_pos_encoder = SinusoidalPositionalEmbedding(300, 0, 50)
        self.audio_cif_pos_encoder = SinusoidalPositionalEmbedding(300, 0, 50)
        self.trans_d_rec = DualTransformer()

        self.mask_vec = nn.Parameter(torch.zeros(config['QueryEncoder']['input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['QueryEncoder']['input_size']).float(), requires_grad=True).cuda()
        self.vocab_size = config['vocab_size']

        self.glove_dim = 300
        self.ctc_fc = nn.Linear(self.glove_dim, self.glove_dim)
        self.ctc_fc2 = nn.Linear(self.glove_dim, self.glove_dim)
        self.fc_comp = nn.Linear(self.glove_dim, 10000+1)
        self.fc_comp_aux = nn.Linear(self.glove_dim, 30064)
        self.predict_ctc = nn.Linear(self.glove_dim, 10000+1)#.requires_grad_(False) # vocab_size

        self.se_encoder = TransformerEncoder(4, self.glove_dim, 6, 0.1, pre_ln=True).requires_grad_(True)

        self.pre_word_con_fc = nn.Linear(self.glove_dim, 300)
        self.pre_audio_con_fc = nn.Linear(self.glove_dim, 300)

        self.pre_audio_sem_gru = DynamicGRU(self.glove_dim, self.glove_dim//2, bidirectional=True, batch_first=True).requires_grad_(True)
        self.sent_ground_fc = nn.Linear(300 * 2, 256)

        from models.modules.cif import CifMiddleware
        self.cif_conv = torch.nn.Conv1d(
            300,
            300,
            5,
            stride=1, padding=int(5 / 2),
            dilation=1, groups=1,
            bias=True, padding_mode='zeros'
        ).cuda()
        self.cif_dense_proj = nn.Linear(300, 300).cuda()
        self.cif_weight_proj = nn.Linear(300, 1).cuda()
        self.cif = CifMiddleware(0.99, 300, 300).cuda().requires_grad_(True)

        self.trans_d_aux = TransformerEncoder(4, self.glove_dim, 6, 0.1, pre_ln=True)
        self.neg_k = 256

        # AVCL module
        self.dropout = nn.Dropout(p=0.5)
        self.R_EASY = config['r_easy']  # 5
        self.R_HARD = config['r_hard']  # 20
        self.m = config['m']
        self.M = config['M']
        self.extend = config['extend']
        self.gradient = None
        self.q_contrast = SniCoLoss_cross()

        self.a_project = nn.Linear(256, 256)
        self.v_project = nn.Linear(256, 256)

    def forward(self, words_feat, audio_feat, audio_feat_wv, words_len, audio_len, audio_len_wv, words_id_len=None, words_len_all=None, epoch=2, neg_words_feat=None, neg_words_len=None, frames_feat=None, frames_len=None, weights=None, words_id=None, words_id_all=None, get_negative=False, **kwargs):

        res = {}
        dropout_rate = 0.1
        ori_audio_len = audio_len

        if kwargs['is_pretrain']:
            # encode audio
            audio_feat = F.dropout(audio_feat, dropout_rate, self.training)
            words_feat = F.dropout(words_feat, dropout_rate, self.training)
            words_len[words_len <= 0] = 1
            words_mask = generate_mask(words_feat, words_len)

            # encode sent
            sent_encoded, words_encoded = self.query_encoder(words_feat, words_len, words_mask)
            neg_sent_encoded = neg_words_feat.sum(dim=2).view(-1, self.neg_k, 300) / (neg_words_len.view(-1, self.neg_k, 1) + 1e-10)

            # Acoustic encoding
            audio_encoded, audio_encoded_wv = self.audio_encoder(audio_feat, audio_feat_wv, audio_len, audio_len_wv)
            audio_len = (audio_len // 4 - 1)  # // 4
            audio_len_wv = (audio_len_wv // 640 - 2)
            audio_len[audio_len <= 0] = 1
            audio_len_wv[audio_len_wv <= 0] = 1
            audio_mask = generate_mask(audio_encoded, audio_len)
            audio_mask_wv = generate_mask(audio_encoded_wv, audio_len_wv)

            res['mask'] = audio_mask


            # Semantic Task -- CTC warm up
            ctc_out = F.relu_(self.ctc_fc2(F.relu_(self.ctc_fc(audio_encoded))))
            ctc_out = self.predict_ctc(ctc_out)

            ctc_out_wv = F.relu_(self.ctc_fc2(F.relu_(self.ctc_fc(audio_encoded_wv))))
            ctc_out_wv = self.predict_ctc(ctc_out_wv)
            # res['ctc_pred'] = ctc_out
            ctc_out = F.log_softmax(ctc_out, dim=-1)
            ctc_out_wv = F.log_softmax(ctc_out_wv, dim=-1)
            # pred = ctc_out.max(dim=-1)[1].detach().cpu().numpy()


            ctc_loss = F.ctc_loss(ctc_out.transpose(0, 1), words_id.cpu(), audio_len.cpu(), words_id_len.cpu(),
                                  blank=10000,
                                  reduction='mean')  # .mean()
            audio_len_wv[audio_len_wv <= 0] = 1
            ctc_loss_wv = F.ctc_loss(ctc_out_wv.transpose(0, 1), words_id.cpu(), audio_len_wv.cpu(),
                                     words_id_len.cpu(),
                                     blank=10000,
                                     reduction='mean')  # .mean()

            # Conciseness Task
            from models.modules.cif import cif_function
            audio_cif, audio_cif_mask, quantity_out = self.cif(audio_encoded, 1 - audio_mask, words_len)
            audio_cif_len = audio_cif_mask.sum(dim=-1)
            audio_cif_len[audio_cif_len <= 0] = 1

            audio_cif_wv, audio_cif_mask_wv, quantity_out_wv = self.cif(audio_encoded_wv, 1 - audio_mask_wv, words_len)
            audio_cif_len_wv = audio_cif_mask_wv.sum(dim=-1)
            audio_cif_len_wv[audio_cif_len_wv <= 0] = 1

            # (cif fast version)
            # conv_out = self.cif_conv(audio_encoded.permute(0, 2, 1)).permute(0, 2, 1)
            # proj_out = self.cif_dense_proj(conv_out)
            # act_proj_out = torch.relu(proj_out)
            # sig_input = self.cif_weight_proj(act_proj_out)
            # cif_weight = torch.sigmoid(sig_input).squeeze(-1)
            # cif_dict = cif_function(audio_encoded, cif_weight, padding_mask=1 - audio_mask)
            # audio_cif, audio_cif_len, quantity_out = cif_dict['cif_out'], cif_dict['cif_lengths'], cif_dict['alpha_sum']
            # audio_cif_len[audio_cif_len <= 0] = 1
            # audio_cif_mask = generate_mask(audio_cif, audio_cif_len)
            #
            # # audio_cif_wv, audio_cif_mask_wv, quantity_out_wv = self.cif(audio_encoded_wv, 1 - audio_mask_wv, words_len_all)
            # # audio_cif_len_wv = audio_cif_mask_wv.sum(dim=-1)
            # conv_out_wv = self.cif_conv(audio_encoded_wv.permute(0, 2, 1)).permute(0, 2, 1)
            # proj_out_wv = self.cif_dense_proj(conv_out_wv)
            # act_proj_out_wv = torch.relu(proj_out_wv)
            # sig_input_wv = self.cif_weight_proj(act_proj_out_wv)
            # cif_weight_wv = torch.sigmoid(sig_input_wv).squeeze(-1)
            # cif_dict_wv = cif_function(audio_encoded_wv, cif_weight_wv, padding_mask=1 - audio_mask_wv)
            # audio_cif_wv, audio_cif_len_wv, quantity_out_wv = cif_dict_wv['cif_out'], cif_dict_wv['cif_lengths'], \
            #                                                   cif_dict_wv['alpha_sum']
            # audio_cif_len_wv[audio_cif_len_wv <= 0] = 1
            # audio_cif_mask_wv = generate_mask(audio_cif_wv, audio_cif_len_wv)

            # Conciseness Task
            quant_loss = F.l1_loss(quantity_out, words_len_all.float())
            quant_loss_wv = F.l1_loss(quantity_out_wv, words_len_all.float())
            quant_loss_tf = torch.abs(quantity_out - quantity_out_wv).mean()


            # Semantic encoding
            audio_encoded2, _ = self.se_encoder(audio_cif, audio_cif_mask.unsqueeze(1).expand(-1, audio_cif.size(1), -1))
            audio_encoded2_wv, _ = self.se_encoder(audio_cif_wv, audio_cif_mask_wv.unsqueeze(1).expand(-1, audio_cif_wv.size(1), -1))

            audio_semantic_enc = audio_encoded2
            audio_semantic_enc_wv = audio_encoded2_wv


            audio_encoded2 = self.pre_audio_sem_gru(audio_encoded2, audio_cif_len)
            audio_encoded2_wv = self.pre_audio_sem_gru(audio_encoded2_wv, audio_cif_len_wv)

            audio_sent_encoded = audio_encoded2.masked_fill(audio_cif_mask.unsqueeze(-1) == 0, 0).sum(dim=1) / (audio_cif_len.unsqueeze(-1) + 1e-10)
            audio_sent_encoded_wv = audio_encoded2_wv.masked_fill(audio_cif_mask_wv.unsqueeze(-1) == 0, 0).sum(
                dim=1) / (audio_cif_len_wv.unsqueeze(-1) + 1e-10)

            audio_sent_encoded = self.pre_audio_con_fc(audio_sent_encoded)
            audio_sent_encoded_wv = self.pre_audio_con_fc(audio_sent_encoded_wv)

            sent_encoded = self.pre_word_con_fc(sent_encoded)
            neg_sent_encoded = self.pre_word_con_fc(neg_sent_encoded)

            # Semantic task -- sentence-level objective
            nce_loss = info_nce(audio_sent_encoded, sent_encoded, neg_sent_encoded, negative_mode='paired')
            nce_loss_wv = info_nce(audio_sent_encoded_wv, sent_encoded, neg_sent_encoded, negative_mode='paired')

            # Semantic task -- word-level objective
            bsz, len, dim = words_feat.size()
            x = torch.zeros([bsz, len+1, dim]).cuda()
            x[:, 0, :] = self.start_vec.clone()  # .cuda()
            x[:, 1:, :] = words_feat
            words_feat = x
            words_mask = generate_mask(words_feat, words_len + 1)
            words_pos = self.word_pos_encoder(words_feat)
            words_feat = self._mask_words(words_feat, words_len, weights=weights) + words_pos
            words_feat1 = words_feat[:, :-1]
            words_id1 = words_id
            words_mask1 = words_mask[:, :-1]

            h = self.trans_d_rec(audio_semantic_enc, audio_cif_mask, words_feat1, words_mask1)
            words_logit = self.fc_comp(h)

            h_wv = self.trans_d_rec(audio_semantic_enc_wv, audio_cif_mask_wv, words_feat1,
                                 words_mask1)
            words_logit_wv = self.fc_comp(h_wv)

            words_logit = words_logit.log_softmax(dim=-1)
            nll_loss2 = cal_nll_loss(words_logit, words_id1, words_mask1)
            nll_loss2 = nll_loss2.mean()

            words_logit_wv = words_logit_wv.log_softmax(dim=-1)
            nll_loss2_wv = cal_nll_loss(words_logit_wv, words_id1, words_mask1)
            nll_loss2_wv = nll_loss2_wv.mean()


            # Conciseness Task -- word predict loss for integrate and fire
            audio_cif_pos = self.audio_cif_pos_encoder(audio_cif)
            audio_cif_ = audio_cif + audio_cif_pos
            words_logit = self.fc_comp_aux(self.trans_d_aux(audio_cif_, audio_cif_mask.unsqueeze(1).expand(-1, audio_cif.size(1), -1))[0])

            words_logit = words_logit.log_softmax(dim=-1)
            words_id_all_ = words_id_all
            audio_cif_mask_ = audio_cif_mask
            if words_logit.size(1) != words_id_all.size(1):
                min_len = min(words_logit.size(1), words_id_all.size(1))
                words_logit = words_logit[:, :min_len, :]
                words_id_all_ = words_id_all[:, :min_len]
                audio_cif_mask_ = audio_cif_mask[:, :min_len]

            nll_loss = cal_nll_loss(words_logit, words_id_all_, audio_cif_mask_)
            nll_loss = nll_loss.mean()


            audio_cif_wv_pos = self.audio_cif_pos_encoder(audio_cif_wv)
            audio_cif_wv_ = audio_cif_wv + audio_cif_wv_pos
            words_logit_wv = self.fc_comp_aux(
                self.trans_d_aux(audio_cif_wv_, audio_cif_mask_wv.unsqueeze(1).expand(-1, audio_cif_wv.size(1), -1))[
                    0])

            words_logit_wv = words_logit_wv.log_softmax(dim=-1)
            words_id_all_ = words_id_all
            audio_cif_mask_wv_ = audio_cif_mask_wv
            if words_logit_wv.size(1) != words_id_all.size(1):
                min_len = min(words_logit_wv.size(1), words_id_all.size(1))
                words_logit_wv = words_logit_wv[:, :min_len, :]
                words_id_all_ = words_id_all[:, :min_len]
                audio_cif_mask_wv_ = audio_cif_mask_wv[:, :min_len]

            nll_loss_wv = cal_nll_loss(words_logit_wv, words_id_all_, audio_cif_mask_wv_)
            nll_loss_wv = nll_loss_wv.mean()

            # Robustness Task
            if audio_semantic_enc_wv.size(1) == audio_semantic_enc.size(1):
                loss_rob = F.l1_loss(audio_semantic_enc_wv, audio_semantic_enc)
            else:
                loss_rob = F.l1_loss(audio_semantic_enc_wv.mean(dim=1), audio_semantic_enc.mean(dim=1))
        #
        # 
            if epoch <= 10:
                # warm-up without quant loss
                loss = (ctc_loss + nce_loss + nll_loss + nll_loss2) + (
                            ctc_loss_wv + nce_loss_wv + nll_loss_wv + nll_loss2_wv) + loss_rob
                res['ctc_loss'] = (ctc_loss)
                res['ctc_loss_wv'] = (ctc_loss_wv)
                res['sent_loss'] = (nce_loss + nce_loss_wv) / 2
                res['quant_loss'] = quant_loss  # (quant_loss + quant_loss_wv) / 2
                res['nar_loss'] = (nll_loss + nll_loss_wv) / 2
                # res['auxwpr_loss'] = nll_loss2
                # res['wpr_loss'] = nll_loss
                res['rec_loss'] = (nll_loss2 + nll_loss2_wv) / 2
                res['rob_loss'] = loss_rob
                res['loss'] = loss

            else:
                loss = (ctc_loss + nce_loss + nll_loss + nll_loss2 + quant_loss) + (ctc_loss_wv + nce_loss_wv + nll_loss_wv + nll_loss2_wv + quant_loss_wv) + loss_rob + quant_loss_tf
                res['ctc_loss'] = (ctc_loss )
                res['ctc_loss_wv'] = (ctc_loss_wv)
                res['sent_loss'] = (nce_loss + nce_loss_wv) / 2
                res['quant_loss'] = quant_loss #(quant_loss + quant_loss_wv) / 2
                res['nar_loss'] = (nll_loss + nll_loss_wv) / 2
                # res['auxwpr_loss'] = nll_loss2
                # res['wpr_loss'] = nll_loss
                res['rec_loss'] = (nll_loss2 + nll_loss2_wv) / 2
                res['rob_loss'] = loss_rob
                res['loss'] = loss
            return res

        frames_feat = F.dropout(frames_feat, dropout_rate, self.training)
        frames_encoded = self.video_encoder(frames_feat)

        audio_feat = F.dropout(audio_feat, dropout_rate, self.training)

        # Acoustic encoding
        audio_encoded, audio_encoded_wv = self.audio_encoder(audio_feat, audio_feat_wv, audio_len, audio_len_wv)
        audio_len = (audio_len // 4 - 1)  # // 4
        audio_len_wv = (audio_len_wv // 640 - 2)
        audio_mask = generate_mask(audio_encoded, audio_len)
        audio_mask_wv = generate_mask(audio_encoded_wv, audio_len_wv)

        res['mask'] = audio_mask
        audio_len[audio_len <= 0] = 1
        from models.modules.cif import cif_function
        audio_cif, audio_cif_mask, quantity_out = self.cif(audio_encoded, 1 - audio_mask, None)
        audio_cif_len = audio_cif_mask.sum(dim=-1)

        audio_cif_wv, audio_cif_mask_wv, quantity_out_wv = self.cif(audio_encoded_wv, 1 - audio_mask_wv, None)
        audio_cif_len_wv = audio_cif_mask_wv.sum(dim=-1)

        # Semantic encoding
        audio_encoded2, _ = self.se_encoder(audio_cif, audio_cif_mask.unsqueeze(1).expand(-1, audio_cif.size(1), -1))
        audio_encoded2_wv, _ = self.se_encoder(audio_cif_wv,
                                               audio_cif_mask_wv.unsqueeze(1).expand(-1, audio_cif_wv.size(1), -1))

        audio_cif_len[audio_cif_len <= 0] = 1
        audio_cif_len_wv[audio_cif_len_wv <= 0] = 1
        audio_encoded2 = self.pre_audio_sem_gru(audio_encoded2, audio_cif_len)
        audio_encoded2_wv = self.pre_audio_sem_gru(audio_encoded2_wv, audio_cif_len_wv)

        if audio_encoded2.size(1) < audio_encoded2_wv.size(1):
            audio_cif_len_final = audio_cif_len
            audio_cif_mask_final = audio_cif_mask
            audio_encoded2_final = torch.cat([audio_encoded2, audio_encoded2_wv[:, :audio_encoded2.size(1)]], dim=-1)
        else:
            audio_cif_len_final = audio_cif_len_wv
            audio_cif_mask_final = audio_cif_mask_wv
            audio_encoded2_final = torch.cat([audio_encoded2[:, :audio_encoded2_wv.size(1)], audio_encoded2_wv], dim=-1)
        audio_encoded2_final = self.sent_ground_fc(audio_encoded2_final)
        fused_h, attn_weight, _, _ = self.fusion(audio_encoded2_final, None, audio_cif_mask_final,
                                              frames_encoded)  # * weight + self.back * (1 - weight))

        # fused_h = frames_encoded


        kwargs['props'] = kwargs['props'].squeeze(0)  # [153,2]
        kwargs['props_graph'] = kwargs['props_graph'].squeeze(0)  # [153,153]


        props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2),
                                             **kwargs)  # [b, 153, 512],  [b, 512, 64, 64],  [b, 1, 64, 64]

        map_h[:, :, kwargs['props'][:, 0], kwargs['props'][:, 1] - 1] = props_h.transpose(1, 2)
        map_h = F.dropout(map_h, dropout_rate, self.training)
        score, frame_score = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask,
                                         props_all=self.prop.props_all_pos, new_props_graph=self.prop.props_graph,
                                         **kwargs)  # [b,153]

        res['score'] = score

        # # contrastive semantic learning
        bsz = score.size(0)
        frame_score = frame_score.detach()
        frame_score_sigmoid = torch.sigmoid(frame_score)
        # # frame_score_sigmoid.requires_grad_(True)
        idx = torch.argsort(score, dim=-1, descending=True)
        props1 = kwargs['props'][idx[:, 0]].contiguous()  # [b, 200, 2]
        pseudo_score = torch.zeros_like(frame_score)
        # pseudo_score_pos, pseudo_score_neg = torch.zeros_like(frame_score), torch.zeros_like(frame_score)
        pseudo_mask = torch.zeros_like(frame_score)
        row_num = props1[:, 1] - props1[:, 0] #+ 1
        # if kwargs['is_training']:
        #    fused_h.register_hook(self.save_gradient)
        for i in range(bsz):
            pseudo_score[i, :] = frame_score_sigmoid[i, :]
            # if  kwargs['is_training'] and self.gradient is not None:
            #     pseudo_score[i, :] = pseudo_score[i, :] #* self.gradient[i, :].sum(-1)
            pseudo_mask[i, props1[i, 0]:props1[i, 1]] = 1
        # inter-frame loss
        pseudo_score_pos = pseudo_score * pseudo_mask
        pseudo_score_neg = pseudo_score * (1 - pseudo_mask)

        num_segments = fused_h.shape[1]
        k_hard = num_segments // self.R_HARD # 3
        k_easy = num_segments // self.R_EASY # 12
        hard_act, hard_bkg, ha_mask, hb_mask, idx_region_inner, idx_region_outer = self.hard_snippets_mining(pseudo_score, frames_encoded, k_hard,
                                                                         row_num)
        easy_act, easy_bkg, ea_mask, eb_mask = self.easy_snippets_mining(pseudo_score_pos, pseudo_score_neg, frames_encoded, k_easy, idx_region_inner, idx_region_outer)
        ea_mask, eb_mask = torch.from_numpy(ea_mask).cuda(), torch.from_numpy(eb_mask).cuda()
        contrast_pairs = {
            'EA': easy_act,
            'EB': easy_bkg,
            'HA': hard_act,
            'HB': hard_bkg
        }
        res['contrast_pairs'] = contrast_pairs

        # nce-based
        contrast_pairs2 = {
            'Q': audio_encoded2_final.masked_fill(audio_cif_mask_final.unsqueeze(-1) == 0, 0).sum(dim=1) / audio_cif_len_final.unsqueeze(-1),
            'EA': easy_act,
            'EB': easy_bkg,
        }
        contrast_fa1 = self.q_contrast(contrast_pairs2)

        # MI-based
        # frames_encoded_project = self.a_project2(F.relu_(self.a_project(frames_encoded)))
        # audio_sent_enc = audio_encoded2_final.masked_fill(audio_cif_mask_final.unsqueeze(-1) == 0, 0).sum(dim=1) / audio_cif_len_final.unsqueeze(-1)
        # audio_sent_enc_project = self.v_project2(F.relu_(self.v_project(audio_sent_enc)))
        # contrast_fa1 = self.batch_video_query_loss(frames_encoded_project, audio_sent_enc_project, ea_mask, eb_mask)
        res['contrast_fa'] = contrast_fa1



        idx = kwargs['neg']
        frames_encoded = frames_encoded[idx]
        #ori_frames_encoded = frames_encoded + 1 - 1
        fused_h, attn_weight, _, _ = self.fusion(audio_encoded2_final, None, audio_cif_mask_final,
                                              frames_encoded)  # * weight + self.back * (1 - weight), attn_weight=None)

        props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2), **kwargs)
        map_h[:, :, kwargs['props'][:, 0], kwargs['props'][:, 1] - 1] = props_h.transpose(1, 2)

        map_h = F.dropout(map_h, dropout_rate, self.training)
        score_z, score_z_rank = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask,
                                            props_all=self.prop.props_all_pos, new_props_graph=self.prop.props_graph,
                                            **kwargs)


        res.update({
            'inter_neg': {
                'neg_score': score_z,
                'weight': None,
            }
        })

        return res

    def generate_word_feat(self, audio_feat, timestamp, time_len):
        props_feats = []
        props_len = []
        bsz, num, _ = timestamp.size()

        for f, p in zip(audio_feat, timestamp):
            for s, e in p:
                s, e = int(s * 127), int(e * 127)
                clip_len = e - s
                #idx = np.linspace(start=0, stop=clip_len - 1, num=16).astype(np.int32)
                try:
                    props_feats.append(f[s:e + 1].mean(dim=0))
                except IndexError:
                    print(f.size(), (s, e))
                    exit(0)
                #props_len.append(props_feats[-1].size(0))
        # print(props_len)
        # exit(0)
        # max_len = max(props_len)
        # for i in range(len(props_feats)):
        #     props_feats[i] = F.pad(props_feats[i], [0, 0, 0, max_len - props_len[i]])

        props_feats = torch.stack(props_feats, 0).view(bsz, num, -1)
        # props_len = torch.from_numpy(np.asarray(props_len).astype(np.int64)).cuda()
        # props_mask = _generate_mask(props_feats, props_len)
        return props_feats#, props_len, None

    def generate_word_feat_layer(self, audio_feat, timestamp, time_len):
        props_feats = []
        props_len = []
        bsz, num, _ = timestamp.size()

        for f, p in zip(audio_feat, timestamp):
            for s, e in p:
                s, e = int(s * 128), int(e * 128)
                clip_len = e - s
                #idx = np.linspace(start=0, stop=clip_len - 1, num=16).astype(np.int32)
                try:
                    props_feats.append(f[:, s:e + 1].mean(dim=1))
                except IndexError:
                    print(f.size(), (s, e))
                    exit(0)
                #props_len.append(props_feats[-1].size(0))

        props_feats = torch.stack(props_feats, 0).view(bsz, num, 6, -1).transpose(1, 2)
        return props_feats#, props_len, None



    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        # token, _ = self.query_encoder(token, None, None)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            # l = min(l, len(p))
            l = min(l, weights.size(1))
            num_masked_words = max(l // 2, 1)
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            p = weights[i, :l].cpu().numpy()

            a = p.sum()
            # # print(p)
            if a == 0:
                p = np.ones([l]) / l
            # if i==33:
            #     print("??")
            # print(l)
            # print(num_masked_words)
            # print(p)
            if l!=len(p):
                print("?")
            choices = np.random.choice(np.arange(1, l+1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        # exit(0)
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1


    def select_topk_embeddings(self, scores, embeddings, k, region_num=None):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]

        k_num = torch.ones_like(region_num) * k
        valid_num = torch.min(region_num, k_num).type(torch.uint8)
        valid_num[valid_num==0] = 1
        for i in range(len(valid_num)):
            now_num = (valid_num[i] - 1).item()
            #valid_idx = idx_topk[i, now_num]
            #idx_topk[i, (now_num+1):] = valid_idx
        mask = torch.zeros_like(scores).cuda()
        bsz = mask.size(0)
        for i in range(bsz):
            mask[i, idx_topk[i, :k]] = 1
        #mask[:, idx_topk[]] = 1
        mask_np = mask.detach().cpu().numpy()
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings, mask

    def easy_snippets_mining(self, actionness, actionness_neg, embeddings, k_easy, idx_in, idx_out):
        select_idx = torch.ones_like(actionness).cuda()
        # select_idx = self.dropout(select_idx)
        # select_idx = (1 - idx_in) * (1 - idx_out)
        #select_idx = self.dropout(select_idx)
        valid_num = select_idx.sum(dim=-1)
        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness_neg, dim=1, keepdim=True)[0] - actionness_neg
        actionness_rev_drop = actionness_rev * select_idx

        easy_act, ea_mask = self.select_topk_embeddings(actionness_drop, embeddings, k_easy, valid_num)
        easy_bkg, eb_mask = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy, valid_num)

        ea_mask = ea_mask.cpu().numpy()
        eb_mask = eb_mask.cpu().numpy()
        dilation_easy_act = ndimage.binary_dilation(ea_mask, structure=np.ones((1, self.extend))).astype(ea_mask.dtype)

        dilation_easy_act = torch.tensor(dilation_easy_act).cuda()
        center_num = dilation_easy_act.sum(dim=-1)
        actionness_center = actionness_drop * dilation_easy_act
        # easy_act, ea_mask = self.select_topk_embeddings(actionness_center, embeddings, k_easy, center_num)

        #dilation_easy_bkg = ndimage.binary_dilation(ea_mask, structure=np.ones((1, self.m))).astype(ea_mask.dtype)

        return easy_act, easy_bkg, ea_mask, eb_mask

    def hard_snippets_mining(self, actionness, embeddings, k_hard, row_num):
        actionness_sum = actionness.sum(dim=-1)
        aness_median = actionness_sum / row_num
        aness_np = actionness.cpu().detach().numpy()
        #aness_median = np.median(aness_np, 1, keepdims=True)
        aness_median = aness_median.view(-1, 1).cpu().detach().numpy()
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        idx_region_inner =  actionness.new_tensor(erosion_m - erosion_M) #actionness.new_tensor(erosion_m - aness_bin)
        aness_region_inner = actionness * idx_region_inner
        inner_num = idx_region_inner.sum(dim=-1)
        hard_act, ha_mask = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard, inner_num)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m) #actionness.new_tensor(aness_bin - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        outer_num = idx_region_outer.sum(dim=-1)
        hard_bkg, hb_mask = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard, outer_num)

        return hard_act, hard_bkg, ha_mask, hb_mask, idx_region_inner, idx_region_outer

    def batch_video_query_loss(self, video, query, pos_mask, neg_mask, measure='JSD'):
        """
            QV-CL module
            Computing the Contrastive Loss between the video and query.
            :param video: video rep (bsz, Lv, dim)
            :param query: query rep (bsz, dim)
            :param match_labels: match labels (bsz, Lv)
            :param mask: mask (bsz, Lv)
            :param measure: estimator of the mutual information
            :return: L_{qv}
        """
        # generate mask
        pos_mask = pos_mask.type(torch.float32)  # (bsz, Lv)
        neg_mask = neg_mask.type(torch.float32) #* mask  # (bsz, Lv)

        # compute scores
        query = query.unsqueeze(2)  # (bsz, dim, 1)
        res = torch.matmul(video, query).squeeze(2)  # (bsz, Lv)

        # computing expectation for the MI between the target moment (positive samples) and query.
        E_pos = self.get_positive_expectation(res * pos_mask, measure, average=False)
        # a1 = torch.sum(E_pos * pos_mask, dim=1)
        # a2 =  (torch.sum(pos_mask, dim=1) + 1e-12)
        E_pos = torch.sum(E_pos * pos_mask, dim=1) / (torch.sum(pos_mask, dim=1) + 1e-12)  # (bsz, )

        # computing expectation for the MI between clips except target moment (negative samples) and query.
        E_neg = self.get_negative_expectation(res * neg_mask, measure, average=False)
        # b1 = torch.sum(E_neg * neg_mask, dim=1)
        # b2 = (torch.sum(neg_mask, dim=1) + 1e-12)
        E_neg = torch.sum(E_neg * neg_mask, dim=1) / (torch.sum(neg_mask, dim=1) + 1e-12)  # (bsz, )

        E = E_neg - E_pos  # (bsz, )
        return torch.mean(E)

    def get_positive_expectation(self, p_samples, measure='JSD', average=True):
        """
        Computes the positive part of a divergence / difference.
        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        log_2 = math.log(2.)
        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = torch.ones_like(p_samples) - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise ValueError('Unknown measurement {}'.format(measure))
        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, measure='JSD', average=True):
        """
        Computes the negative part of a divergence / difference.
        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        log_2 = math.log(2.)
        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'DV':
            Eq = self.log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            raise ValueError('Unknown measurement {}'.format(measure))
        if average:
            return Eq.mean()
        else:
            return Eq

    def log_sum_exp(x, axis=None):
        """
        Log sum exp function
        Args:
            x: Input.
            axis: Axis over which to perform sum.
        Returns:
            torch.Tensor: log sum exp
        """
        x_max = torch.max(x, axis)[0]
        y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
        return y

    def get_modularized_queries(self, encoded_query, query_mask=None):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        if query_mask is not None:
            modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        else:
            modular_attention_scores = F.softmax(modular_attention_scores, dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)

        return modular_queries[:, 0] # (N, D) * 2

    def save_gradient(self, grad):
        self.gradient = grad

    # def save_gradient_test(self, grad): # avoid gradient calculation
    #     self.gradient = None


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


def l2_norm(input, axis):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


# contrast loss
class SniCoLoss_cross(nn.Module):
    def __init__(self):
        super(SniCoLoss_cross, self).__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=1.0):
        q = self.linear1(q.unsqueeze(1)).squeeze(1)
        k = self.linear2(k.unsqueeze(1)).squeeze(1)
        neg = self.linear2(neg)

        q = nn.functional.normalize(q, dim=1) # [bsz, dim]
        k = nn.functional.normalize(k, dim=1) # [bsz, dim]
        neg = neg.permute(0,2,1) # [bsz ,dim ,len]
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        q_refinement = self.NCE(
            contrast_pairs['Q'], # torch.mean(contrast_pairs['Q'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        loss = q_refinement
        return loss


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        # >>> loss = InfoNCE()
        # >>> batch_size, num_negative, embedding_size = 32, 48, 128
        # >>> query = torch.randn(batch_size, embedding_size)
        # >>> positive_key = torch.randn(batch_size, embedding_size)
        # >>> negative_keys = torch.randn(num_negative, embedding_size)
        # >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
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
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class DualTransformer(nn.Module):
    def __init__(self, d_model=300, num_heads=4, num_decoder_layers1=1, num_decoder_layers2=4, dropout=0.1):
        super().__init__()
        self.decoder1 = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout)
        self.decoder2 = TransformerDecoder(num_decoder_layers2, d_model, num_heads, dropout)

    def forward(self, src1, src_mask1, src2, src_mask2, enc_out=None):
        # if enc_out is None:
        #     enc_out = self.decoder2(None, None, src2, src_mask2)
        # out = self.decoder1(enc_out, src_mask2, src1, src_mask1)
        if enc_out is None:
            enc_out = self.decoder1(None, None, src1, src_mask1)
        out = self.decoder2(enc_out, src_mask1, src2, src_mask2)

        return out





def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def forward(self, src, src_mask, tgt, tgt_mask):
        non_pad_src_mask = None if src_mask is None else 1 - src_mask
        non_pad_tgt_mask = None if tgt_mask is None else 1 - tgt_mask

        if src is not None:
            src = src.transpose(0, 1)

        x = tgt.transpose(0, 1)
        for layer in self.decoder_layers:
            x, weight = layer(x, non_pad_tgt_mask,
                              src, non_pad_src_mask,
                              self.buffered_future_mask(x))
        return x.transpose(0, 1)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = MultiheadAttention(d_model, num_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 1)
        self.fc2 = nn.Linear(d_model << 1, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, encoder_out=None, encoder_mask=None, self_attn_mask=None):
        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        if encoder_out is not None:
            res = x
            x, weight = self.encoder_attn(x, encoder_out, encoder_out, encoder_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = res + x
            x = self.encoder_attn_layer_norm(x)

        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x, weight


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    logit = logit.log_softmax(dim=-1)
    # idx:[48, 25] logit:[48, 25, 11743]
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [nb * nw, seq_len]
    smooth_loss = -logit.sum(dim=-1)  # [nb * nw, seq_len]
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        # [nb * nw, seq_len]
        nll_loss = (nll_loss * weights).sum(dim=-1)
    # nll_loss = nll_loss.mean()
    return nll_loss.contiguous()


def map_skip_none(fn, it):
    """
    emulate list(map(fn, it)) but leave None as it is.
    """
    ret = []
    for x in it:
        if x is None:
            ret.append(None)
        else:
            ret.append(fn(x))
    return ret
