import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.weakly_graph.fusion as fusion
import models.weakly_graph.query_encoder as query_encoder
from models.weakly_graph.audio_encoder import AudioEncoder
import models.weakly_graph.query_graph_encoder as query_graph_encoder
import models.weakly_graph.scorer as scorer
import models.weakly_graph.video_encoder as video_encoder
from models.modules import NetVLAD, Filter, Filter2, TanhAttention
from models.modules.graph_attention import GATGraphConv
from models.weakly_graph.prop import SparsePropMaxPool,SparsePropConv
from models.weakly_graph.filter import Filter3, Filter4
from models.modules.dynamic_rnn import DynamicGRU
from models.modules.cross_gate import CrossGate, SelfGate




class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.cross_gate = CrossGate(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 , hidden_dim)

    def forward(self, content, pose):
        #content, pose = self.cross_gate(content, pose)
        #out = content + pose
        out = self.fc(torch.cat([content, pose], -1))
        return out


class WeaklyGraph(nn.Module):

    def __init__(self, config):
        super().__init__()
        #self.clip, self.preprocess = CLIP.load("ViT-B/32", device="cuda", jit=False)
        self.video_encoder = getattr(video_encoder, config['VideoEncoder']['name'])(config['VideoEncoder'])
        # self.video_encoder_attn = getattr(video_encoder, config['VideoEncoderAttn']['name'])(config['VideoEncoderAttn'])
        #self.video_encoder2 = getattr(video_encoder, config['VideoEncoder2']['name'])(config['VideoEncoder2'])
        self.query_encoder = getattr(query_encoder, config['QueryEncoder']['name'])(config['QueryEncoder'])

        # self.audio_vae = SSE()
        self.audio_encoder = AudioEncoder()
        self.query_graph_encoder = getattr(query_graph_encoder, config['QueryGraphEncoder']['name'])(config['QueryGraphEncoder'])
        self.fusion = getattr(fusion, config['Fusion']['name'])(config['Fusion'])
        self.fuse_attn = TanhAttention(512)
        self.fuse_gru = DynamicGRU(512 * 2, 512 // 2,
                                   num_layers=1, bidirectional=True, batch_first=True)
        self.fusion_peer = getattr(fusion, config['Fusion']['name'])(config['Fusion'])
        self.prop = SparsePropMaxPool(config['Fusion']['SparsePropMaxPool'])
        self.scorer = getattr(scorer, config['Scorer']['name'])(config['Scorer'])
        self.back = nn.Parameter(torch.zeros(1, 1, 256), requires_grad=False)
        self.filter_branch = config['filter_branch']
        if self.filter_branch:
            self.filter3 = Filter3(config['Filter'])

        self.layer_norm = nn.LayerNorm(512)

        # contrast
        self.dropout = nn.Dropout(p=0.5)
        self.R_EASY = config['r_easy'] # 5
        self.R_HARD = config['r_hard'] # 20
        self.m = config['m']
        self.M = config['M']
        self.extend = config['extend']

        # self.modular_vector_mapping = nn.Linear(in_features=256, out_features=1, bias=False)

        # audio

        # self.audio_net = base_models.resnet18(modal='audio')
        #self.m = nn.Sigmoid()
        # audio ops
        # self.pooling_a = nn.AdaptiveMaxPool2d((1, 1))
        # self.fc_a_1 = nn.Linear(512, 256)
        # self.fc_a_2 = nn.Linear(256, 256)

        self.gradient = None

        self.q_contrast = SniCoLoss_cross()
        self.word_linear = nn.Linear(300, 256)
        #self.clean_trans = nn.Linear(256, 256)

        # self.stft = spec.STFT(n_fft=160, hop_length=80, window='hann',center=True, pad_mode='constant', output_format='Magnitude')
        # self.mel_sp = spec.MelSpectrogram(sr=16000, n_mels=64)
        # self.power_to_db = torchaudio.transforms.AmplitudeToDB()
        # temporal mask

    def forward(self, frames_feat, frames_len, words_feat, audio_feat, words_len, audio_len, verb_masks, noun_masks, node_roles, rel_edges, spk_id, audio_noise=None, noise=None, notnoise=None, time_mask_feat=None, images=None, text=None, bias=0.0, noisy_feat=None, get_negative=False, **kwargs):
        matrix_mask = torch.tril(torch.ones([32, 32])).cuda()
        matrix_mask = matrix_mask.bool()

        res = {}
        dropout_rate = 0.1
        # 1.add extra dropout
        # frames_feat = F.normalize(frames_feat, dim=-1)
        # audio_feat = F.normalize(audio_feat, dim=-1)

        # clip_adj = torch.empty(64, 64) \
        #     .float().fill_(float(0)).cuda()
        # for i in range(0, 64):
        #     low = i - 3
        #     low = 0 if low < 0 else low
        #     high = i + 3
        #     high = 64 if high > 64 else high
        #     # attn_mask[i, low:high] = 0
        #     clip_adj[i, low:high] = 1


        #frames_feat = F.normalize(frames_feat, dim=-1)
        #frames_feat = (frames_feat - torch.mean(frames_feat, dim=-1, keepdim=True)) / (
        #           torch.std(frames_feat, dim=-1, keepdim=True) + 1e-10)
        #audio_feat = (audio_feat - torch.mean(audio_feat, dim=-1, keepdim=True)) / (torch.std(audio_feat, dim=-1, keepdim=True) + 1e-10)
        #std3 = torch.std(audio_feat)
        #audio_feat = F.normalize(audio_feat, dim=-1)
        frames_feat = F.dropout(frames_feat, dropout_rate, self.training)
        audio_feat = F.dropout(audio_feat, dropout_rate, self.training)

        # if kwargs['is_training']:
        #     noisy_feat = F.dropout(noisy_feat, dropout_rate, self.training)
        # if kwargs['is_pretrain']:
        #     if kwargs['get_noise']:
        #         loss, mag_b, mag_b_recon, mag_ba, x_b_recon, x_ba = self.audio_vae(audio_noise, 'b', notnoise)
        #     else:
        #         loss, x_a_recon = self.audio_vae(audio_feat, 'a')
        #     res['vae_loss'] = loss
        #
        # if kwargs['is_pretrain']:
        #     return res

        # if not kwargs['is_training']:
        #     audio_feat = torch.ones_like(audio_feat).cuda()


        # anchor feat —— clean + noise
        # audio_noise_feat = audio_feat.unsqueeze(1)
        # audio_noise_feat = self.wav_encoder(audio_noise_feat)  # [bsz, 256, 100]
        # audio_encoded = audio_noise_feat.transpose(1, 2)
        #audio_encoded = audio_feat.transpose(1, 2) # [bsz, 100, 256]
        #if kwargs['is_training']:
            #audio_encoded, vqloss = self.audio_encoder(audio_feat)
            # audio_noise_decoded = self.wav_decoder(audio_noise_feat)
            # loss_reconstruction = F.l1_loss(audio_noise_decoded.squeeze(1), audio_feat.squeeze(1))
            # res['loss_recon'] = loss_reconstruction

            # # negative
            # # pure noise
            # noise_feat = noise.unsqueeze(1)
            # noise_feat = self.wav_encoder(noise_feat)  # [bsz, 256, 100]
            #noise_encoded, _  = self.audio_encoder(noise)
            #
            # # positive
            # clean_feat = audio_feat.unsqueeze(1)
            # clean_feat = self.wav_encoder(clean_feat)  # [bsz, 256, 100]
            # clean_encoded = clean_feat.transpose(1, 2)


            # anchor
            # time mask
            # tmp_mask_feat = time_mask_feat.unsqueeze(1)
            # tmp_mask_feat = self.wav_encoder(tmp_mask_feat)  # [bsz, 256, 100]
            # tm_encoded = tmp_mask_feat.transpose(1, 2)
            #print("??")
            # audio_transform = self.clean_trans(audio_encoded)
            # loss_trans = F.l1_loss(audio_transform, noise_encoded)
            # res['loss_trans'] = loss_trans
        # audio_encoded = audio_feat.transpose(1, 2) # [bsz, 100, 256]


        bsz = frames_feat.size(0)
        frames_encoded = self.video_encoder(frames_feat) # [64, 64, 512]
        #frames_encoded = self.feat_encoder(frames_encoded)


        res['sim_loss'] = 0
        res['dcor_loss'] = 0
        #frames_encoded = F.normalize(frames_encoded, -1)
        ori_frames_encoded = frames_encoded + 1 - 1

        frame_len = frames_feat.size(1)

        # visual_mask = torch.ones([bsz, 64]).cuda()
        # prop_mask = torch.ones([bsz, 153]).cuda()
        # audio_len = torch.ones([bsz, 64]).cuda()
        kwargs['props'] = kwargs['props'].squeeze(0)  # [153,2]
        kwargs['props_graph'] = kwargs['props_graph'].squeeze(0)  # [153,153]
        if kwargs['args']['dataset']['erase']:

            # # 1.resnet -- audio pathway
            # audio_feat = audio_feat.unsqueeze(1)
            # a = self.audio_net(audio_feat)
            # a = self.pooling_a(a)
            # a_fea = torch.flatten(a, 1)
            # a = self.fc_a_1(a_fea)
            # a = F.relu_(a)
            # a = self.fc_a_2(a)
            # audio_encoded = a.unsqueeze(1)

            # 2.conv1d
            #audio_feat, speaker_feat = self.fc(audio_feat), self.fc2(audio_feat)
            audio_encoded  = self.audio_encoder(audio_feat)
            #audio_len = audio_len #// 8

            audio_mask = generate_mask(audio_encoded, audio_len)
            audio_mask = None
            ori_audio_encoded = audio_encoded + 1 - 1

            res['loss_dcor'] = None
            # res['vq_loss'] = vqloss
            # res['vqdist'] = vqdist


            # frames_encoded, frames_encoded_cross = self.cross_transformer(ori_frames_encoded, audio_encoded, audio_encoded, None, None)
            # audio_encoded, audio_encoded_cross = self.cross_transformer(audio_encoded, ori_frames_encoded, ori_frames_encoded, None, None)

            # if kwargs['is_training']:
            #
            #     noisy_encoded, noisy_vqloss, noisy_vqdist = self.audio_encoder(noisy_feat)
            #     res['noisy_vqdist'] = noisy_vqdist
            #     #audio_mask = generate_mask(noisy_encoded, audio_len)
            #     ori_noisy_encoded = noisy_encoded + 1 - 1
            #
            #     res['vq_loss'] = res['vq_loss'] + noisy_vqloss
            #
            #     frames_encoded2, frames_encoded2_cross = self.cross_transformer(ori_frames_encoded, ori_noisy_encoded,
            #                                                                   ori_noisy_encoded, None, audio_mask)
            #     noisy_encoded2, noisy_encoded2_cross = self.cross_transformer(ori_noisy_encoded, ori_frames_encoded,
            #                                                                 ori_frames_encoded, audio_mask, None)
            #     fused_h_noisy, _, _ = self.fusion(noisy_encoded2, None, audio_mask,
            #                                       frames_encoded2)  # * weight + self.back * (1 - weight))
            #     fused_h_noisy= F.dropout(fused_h_noisy, dropout_rate, self.training)
            #     fused_h_noisy = self.clip_graph(fused_h_noisy, clip_adj)
            #     props_h_noisy, map_h_noisy, map_mask_noisy = self.prop(fused_h_noisy.transpose(1, 2),
            #                                          **kwargs)  # [b, 153, 512],  [b, 512, 64, 64],  [b, 1, 64, 64]
            #     score_noisy, _ = self.scorer(props_h=props_h_noisy, map_h=map_h_noisy, map_mask=map_mask_noisy,
            #                                      props_all=self.prop.props_all_pos,
            #                                      new_props_graph=self.prop.props_graph, **kwargs)  # [b,153]
            #     res['score_noisy'] = score_noisy
            #
            #     frame_score_noisy = score_noisy[:, :64]
            #     # res['props_bag_score'] = props_bag_score
            #     # frame_score.requires_grad_(True)
            #     frame_score_sigmoid = torch.sigmoid(frame_score_noisy)
            #     # frame_score_sigmoid.requires_grad_(True)
            #     idx = torch.argsort(score_noisy, dim=-1, descending=True)
            #     props1 = kwargs['props'][idx[:, 0]].contiguous()  # [b, 200, 2]
            #     pseudo_score_noisy = torch.zeros_like(frame_score_noisy)
            #     pseudo_mask_noisy = torch.zeros_like(frame_score_noisy)
            #     row_num = props1[:, 1] - props1[:, 0]  # + 1
            #     if kwargs['is_training']:
            #         fused_h_noisy.register_hook(self.save_gradient)
            #     for i in range(bsz):
            #         pseudo_score_noisy[i, props1[i, 0]:props1[i, 1]] = frame_score_sigmoid[i, props1[i, 0]:props1[i, 1]]
            #         if kwargs['is_training'] and self.gradient is not None:
            #             pseudo_score_noisy[i, :] = pseudo_score_noisy[i, :]  # * self.gradient[i, :].sum(-1)
            #         pseudo_mask_noisy[i, props1[i, 0]:props1[i, 1]] = 1
            #     # # contrast loss
            #     # inter-frame loss
            #     num_segments = fused_h_noisy.shape[1]
            #     k_hard = num_segments // self.R_HARD
            #     k_easy = num_segments // self.R_EASY
            #     hard_act, hard_bkg, ha_mask, hb_mask, idx_region_inner, idx_region_outer = self.hard_snippets_mining(
            #         pseudo_score_noisy, frames_encoded2_cross, k_hard,
            #         row_num)
            #     easy_act, easy_bkg, ea_mask, eb_mask = self.easy_snippets_mining(pseudo_score_noisy, frames_encoded2_cross,
            #                                                                      k_easy, idx_region_inner,
            #                                                                      idx_region_outer)
            #     # ea_mask_np = ea_mask.detach().cpu().numpy()
            #     # eb_mask_np = eb_mask.detach().cpu().numpy()
            #     # ha_mask_np = ha_mask.detach().cpu().numpy()
            #     # hb_mask_np = hb_mask.detach().cpu().numpy()
            #     contrast_pairs = {
            #         'EA': easy_act,
            #         'EB': easy_bkg,
            #         'HA': hard_act,
            #         'HB': hard_bkg
            #     }
            #     res['contrast_pairs_noisy'] = contrast_pairs
            #
            #     contrast_pairs2 = {
            #         'Q': noisy_encoded2_cross,
            #         'EA': easy_act,
            #         'EB': easy_bkg,
            #     }
            #     # res['contrast_pairs2'] = contrast_pairs2
            #     contrast_fa1 = self.q_contrast(contrast_pairs2)
            #     res['contrast_fa_noisy'] = contrast_fa1
            res['filter_weight'] = None
            res['filter_att'] = None
            res['filter_weight_o'] = None
            # weight, att, weight_o = self.filter3(frames_encoded, audio_encoded, None, None,
            #                                      **kwargs)  # [b,v_len 64,1]

            fused_h, attn_weight, _= self.fusion(audio_encoded, None, audio_mask, frames_encoded)# * weight + self.back * (1 - weight))
            res['frame_feat'] = fused_h


            if kwargs['is_training']:
                # pure noise branch
                #with torch.no_grad():
                # noise
                # frames_encoded_noise, _ = self.cross_transformer(ori_frames_encoded, noise_encoded,
                #                                                   noise_encoded, None, None)
                # noise_encoded, _ = self.cross_transformer(noise_encoded, ori_frames_encoded,
                #                                           ori_frames_encoded, None, None)

                # fused_h_noise, _, _ = self.fusion(noise_encoded, None, None, frames_encoded_noise)
                # props_h_noise, map_h_noise, map_mask_noise = self.prop(fused_h_noise.transpose(1, 2),
                #                                      **kwargs)  # [b, 153, 512],  [b, 512, 64, 64],  [b, 1, 64, 64]
                # score_noise, _ = self.scorer(props_h=props_h_noise, map_h=map_h_noise, map_mask=map_mask_noise,
                #                                  props_all=self.prop.props_all_pos,
                #                                  new_props_graph=self.prop.props_graph, **kwargs)  # [b,153]

                # # clean
                # frames_encoded_clean, _ = self.cross_transformer(ori_frames_encoded, clean_encoded,
                #                                                  clean_encoded, None, None)
                # clean_encoded, _ = self.cross_transformer(clean_encoded, ori_frames_encoded,
                #                                           ori_frames_encoded, None, None)
                #
                # fused_h_clean, _, _ = self.fusion(clean_encoded, None, None, frames_encoded_clean)
                # props_h_clean, map_h_clean, map_mask_clean = self.prop(fused_h_clean.transpose(1, 2),
                #                                                        **kwargs)  # [b, 153, 512],  [b, 512, 64, 64],  [b, 1, 64, 64]
                # score_clean, _ = self.scorer(props_h=props_h_clean, map_h=map_h_clean, map_mask=map_mask_clean,
                #                              props_all=self.prop.props_all_pos,
                #                              new_props_graph=self.prop.props_graph, **kwargs)  # [b,153]

                    # time mask
                    # frames_encoded_tm, _ = self.cross_transformer(ori_frames_encoded, tm_encoded,
                    #                                                  tm_encoded, None, None)
                    # tm_encoded, _ = self.cross_transformer(tm_encoded, ori_frames_encoded,
                    #                                           ori_frames_encoded, None, None)
                    #
                    # fused_h_tm, _, _ = self.fusion(tm_encoded, None, None, frames_encoded_tm)
                    # props_h_tm, map_h_tm, map_mask_tm = self.prop(fused_h_tm.transpose(1, 2),
                    #                                                        **kwargs)  # [b, 153, 512],  [b, 512, 64, 64],  [b, 1, 64, 64]
                    # score_tm, _ = self.scorer(props_h=props_h_tm, map_h=map_h_tm, map_mask=map_mask_tm,
                    #                              props_all=self.prop.props_all_pos,
                    #                              new_props_graph=self.prop.props_graph, **kwargs)  # [b,153]
                res['score_noise'] = None
                # res['score_clean'] = score_clean
                res['erase_tri_loss_verb'] = 0
                res['erase_tri_loss_noun'] = 0
                res['erase_tri_loss'] = 0
                #torch.cuda.empty_cache()


        else:
            if 'is_training' not in kwargs:
                # frames_feat2 = frames_feat
                # words_feat2 = words_feat
                words_mask = words_mask2 = generate_mask(words_feat, words_len)  # [b,20]
            if not kwargs['ensemble']:
                words_mask = generate_mask(words_feat, words_len)  # [b,20]

            words_encoded = self.query_encoder(words_feat, words_len, words_mask)  # [b,20,512]
            if self.filter_branch:
                weight, _ , _= self.filter3(frames_encoded, words_encoded, words_len, words_mask, **kwargs)  # [b,v_len 64,1]
                fused_h, attn_weight, fused_txt= self.fusion(words_encoded, words_len, words_mask,
                                                              frames_encoded * weight + self.back * (1 - weight))
                res['attn_weight'] = attn_weight
            else:
                fused_h, attn_weight, fused_txt= self.fusion(words_encoded, words_len, words_mask, frames_encoded)


        fused_h = F.dropout(fused_h, dropout_rate, self.training)
        # fused_h = self.clip_graph(fused_h, clip_adj)
        props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2), **kwargs)  # [b, 153, 512],  [b, 512, 64, 64],  [b, 1, 64, 64]
        score, frame_score = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask, props_all=self.prop.props_all_pos, new_props_graph=self.prop.props_graph, **kwargs)  # [b,153]

        res['score'] = score
        #frame_score = score[:, :64]
        res['self_score'] = frame_score
        #res['props_bag_score'] = props_bag_score
        # frame_score.requires_grad_(True)
        # frame_score_sigmoid = torch.sigmoid(frame_score)
        # # frame_score_sigmoid.requires_grad_(True)
        # idx = torch.argsort(score, dim=-1, descending=True)
        # props1 = kwargs['props'][idx[:, 0]].contiguous()  # [b, 200, 2]
        # pseudo_score = torch.zeros_like(frame_score)
        # pseudo_mask = torch.zeros_like(frame_score)
        # row_num = props1[:, 1] - props1[:, 0] #+ 1
        # if kwargs['is_training']:
        #    fused_h.register_hook(self.save_gradient)
        # for i in range(bsz):
        #     pseudo_score[i, props1[i, 0]:props1[i, 1]] = frame_score_sigmoid[i, props1[i, 0]:props1[i, 1]]
        #     if  kwargs['is_training'] and self.gradient is not None:
        #         pseudo_score[i, :] = pseudo_score[i, :] #* self.gradient[i, :].sum(-1)
        #     pseudo_mask[i, props1[i, 0]:props1[i, 1]] = 1
        # # contrast loss
        # inter-frame loss
        '''
        num_segments = fused_h.shape[1]
        k_hard = num_segments // self.R_HARD
        k_easy = num_segments // self.R_EASY
        hard_act, hard_bkg, ha_mask, hb_mask, idx_region_inner, idx_region_outer = self.hard_snippets_mining(pseudo_score, frames_encoded_cross, k_hard,
                                                                         row_num)
        easy_act, easy_bkg, ea_mask, eb_mask = self.easy_snippets_mining(pseudo_score, frames_encoded_cross, k_easy, idx_region_inner, idx_region_outer)
        # ea_mask_np = ea_mask.detach().cpu().numpy()
        # eb_mask_np = eb_mask.detach().cpu().numpy()
        # ha_mask_np = ha_mask.detach().cpu().numpy()
        # hb_mask_np = hb_mask.detach().cpu().numpy()
        contrast_pairs = {
            'EA': easy_act,
            'EB': easy_bkg,
            'HA': hard_act,
            'HB': hard_bkg
        }
        res['contrast_pairs'] = contrast_pairs

        contrast_pairs2 = {
            'Q': audio_encoded_cross,
            'EA': easy_act,
            'EB': easy_bkg,
        }
        #res['contrast_pairs2'] = contrast_pairs2
        contrast_fa1 = self.q_contrast(contrast_pairs2)
        res['contrast_fa'] = contrast_fa1
        '''

        if get_negative:
            if 'is_training' not in kwargs or True:
                attn_weight_ori = None
            bsz = frames_feat.size(0)
            # suppress
            if self.filter_branch:

                #sup_frames_encoded = ori_frames_encoded * (1 - weight)
                # fused_h, _, _ = self.fusion(audio_encoded, None, None,
                #                             frames_encoded * (1 - weight) + self.back * weight, attn_weight=attn_weight_ori)


                '''
                fused_h_verb, _, _ = self.fusion(verb_encoded, verb_lens, verb_masks,
                                            frames_encoded * (1 - weight_verb) + self.back * weight_verb,
                                            attn_weight=attn_weight_ori)

                fused_h_noun, _, _ = self.fusion(noun_encoded, noun_lens, noun_masks,
                                                 frames_encoded * (1 - weight_noun) + self.back * weight_noun,
                                                 attn_weight=attn_weight_ori)

                fused_h_sent, _, _ = self.fusion(sent_encoded.unsqueeze(1), None, None,
                                                 frames_encoded * (1 - weight_sent) + self.back * weight_sent,
                                                 attn_weight=attn_weight_ori)

                fused_h_graph = (fused_h_verb + fused_h_noun + fused_h_sent) / 3

                fused_h = fused_h * self.erase_weight_graph + fused_h_graph * (1 - self.erase_weight_graph)
                '''


                # fused_h = F.dropout(fused_h, 0.1, self.training)
                # # fused_h_graph = F.dropout(fused_h_graph, dropout_rate, self.training)
                # props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2), **kwargs)
                # score_neg, _ = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask, props_all=self.prop.props_all_pos, new_props_graph=self.prop.props_graph, **kwargs)


                # props_h_graph, map_h_graph, map_mask_graph = self.prop(fused_h_graph.transpose(1, 2), **kwargs)
                # score_neg_graph = self.scorer(props_h=props_h_graph, map_h=map_h_graph, map_mask=map_mask_graph, **kwargs)
                #
                # score_neg = score_neg * self.erase_weight_graph_suppress + (1-self.erase_weight_graph_suppress) * score_neg_graph

                res.update({
                    'intra_neg': {
                        'weight': 0,
                        'weight_graph':0,
                        'neg_score': score, # score_neg
                    }
                })


                # res.update({
                #     'intra_neg': {
                #         'weight': None,
                #         'neg_score': None,
                #     }
                # })
            else:
                res.update({
                    'intra_neg': {
                        'weight': None,
                        'neg_score': None,
                    }
                })

            # negative
            idx = kwargs['neg']
            frames_encoded = ori_frames_encoded[idx]
            ori_frames_encoded = frames_encoded + 1 - 1
            if self.filter_branch:
                # weight, att, weight_o = self.filter3(frames_encoded, audio_encoded, None, None,
                #                                      **kwargs)  # [b,v_len 64,1]

                # frames_encoded, _ = self.cross_transformer(frames_encoded, ori_audio_encoded,
                #                                                               ori_audio_encoded, None, audio_mask)
                # audio_encoded, _ = self.cross_transformer(ori_audio_encoded, ori_frames_encoded,
                #                                                             ori_frames_encoded, audio_mask, None)

                fused_h, attn_weight, _= self.fusion(audio_encoded, None, audio_mask,
                                                      frames_encoded)# * weight + self.back * (1 - weight), attn_weight=None)


            else:
                fused_h, _, _= self.fusion(words_encoded, words_len, words_mask, frames_encoded)

            fused_h = F.dropout(fused_h, dropout_rate, self.training)
            #fused_h = self.clip_graph(fused_h, clip_adj)
            props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2), **kwargs)

            score_z, score_z_rank = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask, props_all=self.prop.props_all_pos, new_props_graph=self.prop.props_graph,  **kwargs)

            # if kwargs['is_training']:
            #     frames_encoded2, _ = self.cross_transformer(ori_frames_encoded, ori_noisy_encoded,
            #                                                ori_noisy_encoded, None, audio_mask)
            #     noisy_encoded2, _ = self.cross_transformer(ori_noisy_encoded, ori_frames_encoded,
            #                                               ori_frames_encoded, audio_mask, None)
            #
            #     fused_h_noisy, attn_weight, _ = self.fusion(noisy_encoded2, None, audio_mask,
            #                                           frames_encoded2)  # * weight + self.back * (1 - weight), attn_weight=None)
            #     fused_h_noisy = F.dropout(fused_h_noisy, dropout_rate, self.training)
            #     fused_h_noisy = self.clip_graph(fused_h_noisy, clip_adj)
            #     props_h_noisy, map_h_noisy, map_mask_noisy = self.prop(fused_h_noisy.transpose(1, 2), **kwargs)
            #
            #     score_z_noisy, score_z_rank_noisy = self.scorer(props_h=props_h_noisy, map_h=map_h_noisy, map_mask=map_mask_noisy,
            #                                         props_all=self.prop.props_all_pos,
            #                                         new_props_graph=self.prop.props_graph, **kwargs)
            #     res.update({
            #         'noisy_neg': {
            #             'neg_score': score_z_noisy,
            #
            #         }
            #     })

            res.update({
                    'inter_neg': {
                        'neg_score': score_z,
                        'self_neg_score':score_z_rank,
                        'neg_prob_matrix': None,
                        # 'fused_score': fused_score,
                        # 'self_fused_score':self_fused_score,
                        'weight':None,
                        #'props_bag_score':props_bag_score,
                        #'boundary_score':boundary_neg_score
                    }
                })

        return res

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

    def easy_snippets_mining(self, actionness, embeddings, k_easy, idx_in, idx_out):
        #select_idx = torch.ones_like(actionness).cuda()
        #select_idx = self.dropout(select_idx)
        select_idx = (1 - idx_in) * (1 - idx_out)
        #select_idx = self.dropout(select_idx)
        valid_num = select_idx.sum(dim=-1)
        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act, ea_mask = self.select_topk_embeddings(actionness_drop, embeddings, k_easy // 2, valid_num)
        easy_bkg, eb_mask = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy, valid_num)

        ea_mask = ea_mask.cpu().numpy()
        dilation_easy_act = ndimage.binary_dilation(ea_mask, structure=np.ones((1, self.extend))).astype(ea_mask.dtype)

        dilation_easy_act = torch.tensor(dilation_easy_act).cuda()
        center_num = dilation_easy_act.sum(dim=-1)
        actionness_center = actionness_drop * dilation_easy_act
        easy_act, ea_mask = self.select_topk_embeddings(actionness_center, embeddings, k_easy, center_num)

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
        idx_region_inner = actionness.new_tensor(erosion_m - aness_bin) # actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        inner_num = idx_region_inner.sum(dim=-1)
        hard_act, ha_mask = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard, inner_num)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(aness_bin - dilation_m)#actionness.new_tensor(dilation_M - dilation_m)
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


def generate_mask_erase(x, x_len, erase_word_pos_gt):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for i, l in enumerate(x_len):
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
            for pos in erase_word_pos_gt[i, :]:
                mask[-1][pos.long()] = 0
        mask = torch.stack(mask, 0)
    return mask


def rand_frame_mask(x, rate=0.25, con=16):  # 按segment mask
    b, seq, d = x.size()
    # mask = torch.ones(n, d) * x.mean()
    mask = torch.zeros(con, d)
    n = int(seq * rate) // con
    x = x.detach().cpu().numpy()
    for i in range(b):
        mask_ids = np.random.choice(seq // con, n)  # 有放回采样
        for s in mask_ids:
            x[i, np.arange(s * con, s * con + con), :] = mask
    return torch.from_numpy(x).cuda()


def rand_mask(x, x_len, rate=0.2):
    b, seq, d = x.size()
    n = int(seq * rate)
    mask = torch.ones(n, d) * x.mean()
    # mask = torch.zeros(n, d)
    x = x.detach().cpu().numpy()
    mask_matrix = torch.ones(b, seq)
    for i in range(b):
        mask_id = np.random.choice(seq, n)  # 有放回采样
        x[i, mask_id, :] = mask
        mask_matrix[i, mask_id] = 0
        mask_matrix[i, x_len[i]:] = 0
    return torch.from_numpy(x).cuda(), mask_matrix.cuda()


def rand_downsample(x, rate=0.2):  # 256/320
    b, seq, d = x.size()
    n = seq - int(seq * rate)
    x = x.detach().cpu().numpy()
    y = np.zeros((b, n, d)).astype(np.float32)
    for i in range(b):
        choice = np.random.choice(seq, n)  # 有放回采样
        y[i] = x[i, choice]
    return torch.from_numpy(y).cuda()


def l2_norm(input, axis):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = nn.Conv1d(2 * dim, dim, 1, 1)


    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = output.transpose(1, 2)
        output = self.conv1d(output)
        output = output.transpose(1, 2)
        return output


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = nn.Conv1d(dim, 1, 1, 1)


    def forward(self, x, mask):
        # compute logits
        x = x.transpose(1, 2)
        logits = self.conv1d(x)
        logits = logits.transpose(1, 2)
        logits = logits.squeeze(2)
        if mask is not None:
            logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss

def pairwise_distances(x, y=None):
    """
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between
        x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        source:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return dist


def pairwise_dist(x):
    #x should be two dimensional
    eps = 1e-9
    instances_norm = torch.sum(x**2, -1).reshape((-1,1))
    output = -2*torch.mm(x, x.t()) + instances_norm + instances_norm.t()
    return torch.sqrt(output.clamp(min=0) + eps)


def recon_criterion(x_recon, x=None, notnoise=None, cross=False):
    if cross:
        # Pure noise losses: enforce the output of pure noises to be zeros
        return torch.sum(notnoise.unsqueeze(1).unsqueeze(2) * (x_recon ** 2)) \
                / (torch.sum(notnoise) * x_recon.shape[1] * x_recon.shape[2] + 1e-10)


    if notnoise is not None:
        # Don't compute reonstruction losses for pure noises
        return torch.sum(notnoise.unsqueeze(1).unsqueeze(2) * \
                            ((x_recon ** 0.5 - x ** 0.5) ** 2)) \
                / (torch.sum(notnoise) * x_recon.shape[1] * x_recon.shape[2] + 1e-10)
    return torch.mean(((x_recon + 1e-10) ** 0.5 - (x + 1e-10) ** 0.5) ** 2)


def dcor(x, y):
    eps = 1e-9
    m,_ = x.shape
    assert len(x.shape) == 2
    assert len(y.shape) == 2

    dx = pairwise_dist(x)
    dy = pairwise_dist(y)

    dx_m = dx - dx.mean(dim=0)[None, :] - dx.mean(dim=1)[:, None] + dx.mean()
    dy_m = dy - dy.mean(dim=0)[None, :] - dy.mean(dim=1)[:, None] + dy.mean()

    dcov2_xy = (dx_m * dy_m).sum()/float(m * m)
    dcov2_xx = (dx_m * dx_m).sum()/float(m * m)
    dcov2_yy = (dy_m * dy_m).sum()/float(m * m)

    dcor = torch.sqrt(dcov2_xy)/torch.sqrt((torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy)).clamp(min=0) + eps)

    return dcor


def build_position_encoding(position_emb, HIDDEN_SIZE, num_clips):
    N_steps = HIDDEN_SIZE // 2
    if position_emb in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_emb in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps, num_clips)
    else:
        raise ValueError(f"not supported {position_emb}")

    return position_embedding

import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):

        assert mask is not None
        not_mask = mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256, num_clips=32):
        super().__init__()
        self.row_embed = nn.Embedding(num_clips, num_pos_feats)
        self.col_embed = nn.Embedding(num_clips, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, input):
        x = input
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(x)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


# contrast loss
class SniCoLoss_cross(nn.Module):
    def __init__(self):
        super(SniCoLoss_cross, self).__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
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
            torch.mean(contrast_pairs['Q'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        loss = q_refinement
        return loss


if __name__ == '__main__':
    # rand_mask(torch.randn(2, 64, 512))
    weight = torch.rand([2, 64])
    print(weight)
    frame_topk = torch.argsort(weight, dim=-1, descending=True)
    n = int(64 * 0.2)
    # make the top 20% frame feature unchanged
    frame_topk = frame_topk[:,:n]
    for i in range(frame_topk.size(0)):
        for j in frame_topk[i]:
            weight[i][j] = 1.0
    # weight[frame_topk] = 1.0
    print(weight)
