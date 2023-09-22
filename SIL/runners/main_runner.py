import collections
import logging
import random
import os, json
import nltk
import numpy as np
from torch import nn
import torch
from fairseq.utils import move_to_cuda
from gensim.utils import tokenize

from utils import AverageMeter, TimeMeter, top_1_metric, top_n_metric, calculate_IoU_batch2, load_json, calculate_IoU

import dataset as da
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from models.weakly_graph.loss import weakly_supervised_loss, weakly_supervised_loss_new
from copy import copy


import time
import torch.nn.functional as F
# from pytorch_metric_learning.miners import TripletMarginMiner


class MainRunner:
    def __init__(self, args):
        self.stage = args['train']['stage']  # 'ground' 'pre'
        self.args = args
        self._build_dataset()
        self.args['model']['vocab_size'] = 12000 # self.train_set.vocab_size

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

        self.num_clips = args['dataset']['max_num_frames'] // args['dataset']['target_stride']
        self.score1 = -1
        self.score5 = -1
        self.criterion = nn.BCEWithLogitsLoss()



    def train(self):

        def score(logger1):
            return 0.7 * logger1['IoU@0.7'].avg + 0.2 * logger1['IoU@0.5'].avg + 0.1 * logger1['IoU@0.3'].avg

        # for idx in [10, 20, 30, 40, 50]:
        #     self._load_model('/home/wangye/wangye2/wangye/TIP2021-erase/checkpoints/activitynet/weak/full/oriresult/pre-{}.pt'.format(idx))# model_seed44_lr0.00080_topK20-0.6739_0.5042_0.3067_0.9407_0.8358_0.6685-1.pt')
        #     bias=0.0
        #     print(idx)
        #     sc_1, s1 = self.eval(top_n=1, dataloader=self.test_loader, bias=bias, args=self.args)
        #     print("-----------")
        # exit(0)
        # self._load_model(
        #     '/home/wangye/wangye2/wangye/TIP2021-erase/checkpoints/activitynet/weak/full/oriresult/prefull2-25.pt')


        # self._load_model(
        #     '/home/wangye/wangye2/wangye/TIP2021-erase/checkpoints/activitynet/weak/full/oriresult/960scl+ctc-linearp-46.pt')# 960-scl+wr+ctc+ar-stage2-24.pt #960-scl+wr+ctc+ar-47.pt')#newae-aux2_stage2_rec20.pt') # tmp3-4.pt newae-aux2_stage2_rec20.pt #460_rec2-13 rec2-38 model_seed44_lr0.00080_topK20-0.6739_0.5042_0.3067_0.9407_0.8358_0.6685-1.pt')
        # self._load_model('/home/wangye/wangye2/wangye/TIP2021-erase/checkpoints/activitynet/weak/full/oriresult/tmp-3.pt')

        # self._load_model(
        #     '/home1/lihaoyuan/wangye/TIP2021-erase/checkpoints/activitynet/weak/ncontrast+noise/tmp-zz-2.pt')
        #self._load_model('/home1/yeshangwei/wangye/TIP2021-erase/checkpoints/activitynet/weak/full/oriresult/asr-68.pt') # memory-80

        clips = torch.zeros([528]).cuda()
        self.args['clips'] = clips
        for bias in [0.0]:
            logging.info('bias = {}.'.format(bias))

            for epoch in range(1, self.args['train']['max_num_epochs'] + 1):
                torch.cuda.empty_cache()
                logging.info('Start Epoch {}'.format(epoch))
                self.model_saved_path = self.args['train']['model_saved_path']
                os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)

                if self.stage == 'pre':
                    self._pretrain_one_epoch(epoch, bias=bias, args=self.args, is_pretrain=True, get_noise=False, is_training=True)
                    save_path = os.path.join(self.model_saved_path, 'pre-%d.pt' % (epoch))
                elif self.stage == 'ground':
                    self._train_one_epoch(epoch, bias=bias, args=self.args, is_pretrain=False, get_noise=False,
                                          is_training=True)
                    save_path = os.path.join(self.model_saved_path, 'ground-%d.pt' % (epoch))

                self._save_model(save_path)

                # sc_1, s1 = self.eval(top_n=1, dataloader=self.test_loader,bias=bias, args=self.args)
                # sc_5, s5 = self.eval(dataloader=self.test_loader, bias=bias, args=self.args)
                # if sc_1['IoU@0.7'].avg < 0.09 and epoch == 1:
                #      break

                # if  sc_1['IoU@0.5'].avg > 0.3040 and sc_1['IoU@0.3'].avg > 0.5030 and sc_5['IoU@0.5'].avg > 0.63 and sc_5['IoU@0.3'].avg > 0.79:
                #      # save_path = os.path.join(self.model_saved_path, 'tmp-%s.pt' % (str(self.args['train']['lr']) + '-' + str(self.args['seed']) + '-' + s1 + '-' + s5))
                #      save_path = os.path.join(self.model_saved_path,
                #                               'model_seed%d_lr%.5f_topK%d-%.4f_%.4f_%.4f_%.4f_%.4f_%.4f-%d.pt' %
                #                               (self.args['seed'], self.args['train']['lr'], self.args['train']['topK'],
                #                                sc_1['IoU@0.1'].avg, sc_1['IoU@0.3'].avg, sc_1['IoU@0.5'].avg,sc_5['IoU@0.1'].avg, sc_5['IoU@0.3'].avg, sc_5['IoU@0.5'].avg,  epoch,))
                #      self._save_model(save_path)

                logging.info('=' * 60)
        logging.info('Done.')

    def _pretrain_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)

            # msg = 'Epoch {}, Batch {},  '.format(epoch, bid)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            logging.info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())
        if self.args['dataset']['dataset'] == 'ActivityNetGraph':
            num_cands = 153  #213 #153 # 1552 153 1134 134 155 153  143  111  141
            fp = open('append.log', encoding='utf8', mode='a')
        else:
            if self.args['dataset']['dataset'] == 'CharadesSTA':
                num_cands = 528
            else:
                num_cands = 200
            fp = open('append2.log', encoding='utf8', mode='a')

        for bid, batch in enumerate(self.new_train_loader, 1):
            self.model.train()
            torch.cuda.empty_cache()


            b = len(batch['raw'])
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            net_input['tgt'] = False
            net_input['epoch'] = epoch

            output = self.model(**net_input, get_negative=True, **kwargs)
            torch.cuda.empty_cache()

            # src_low_feat = output['low_feat']#.contiguous().view(-1, 300)
            # src_high_feat = output['high_feat']
            # src_mask = output['mask']
            loss = output['loss']

            loss_ctc = output['ctc_loss']
            loss_ctc_wv = output['ctc_loss_wv']
            loss_quant = output['quant_loss']

            # loss_wpr = output['wpr_loss']
            # loss_aux = output['auxwpr_loss']
            loss_nar = output['nar_loss']
            loss_sent = output['sent_loss']
            loss_rec = output['rec_loss']
            loss_rob = output['rob_loss']

            '''
            # tgt_iter = iter(self.train_loader)

            tgt_batch = next(tgt_iter)
            tgt_net_input = move_to_cuda(tgt_batch['net_input'])
            tgt_net_input['tgt'] = True
            output = self.model(**tgt_net_input, get_negative=True, **kwargs)
            torch.cuda.empty_cache()

            # tgt_ctc_out = output['ctc_pred']
            # ent_loss = entropy(tgt_ctc_out)
            # loss = loss + 0.0001 * ent_loss


            tgt_low_feat = output['low_feat']#.contiguous().view(-1, 300)
            tgt_high_feat = output['high_feat']
            tgt_mask = output['mask']
            
            # src_valid_feat = src_low_feat.masked_select(src_mask.view(-1, 1)==1)
            # tgt_valid_feat = tgt_low_feat.masked_select(tgt_mask.view(-1, 1)==1)
            # src_valid_feat = src_valid_feat.view(-1, 300)
            # tgt_valid_feat = tgt_valid_feat.view(-1, 300)
            # b = src_mask.sum(dim=-1).unsqueeze(-1) + 1e-10
            # a = src_low_feat.masked_fill(src_mask.unsqueeze(-1)==0, 0).sum(dim=1)
            src_low_pool_feat = src_low_feat.masked_fill(src_mask.unsqueeze(-1)==0, 0).sum(dim=1) / src_mask.sum(dim=-1).unsqueeze(-1) + 1e-10
            tgt_low_pool_feat = tgt_low_feat.masked_fill(tgt_mask.unsqueeze(-1) == 0, 0).sum(dim=1) / tgt_mask.sum(
                dim=-1).unsqueeze(-1) + 1e-10

            src_high_pool_feat = src_high_feat.masked_fill(src_mask.unsqueeze(-1) == 0, 0).sum(dim=1) / src_mask.sum(
                dim=-1).unsqueeze(-1) + 1e-10
            tgt_high_pool_feat = tgt_high_feat.masked_fill(tgt_mask.unsqueeze(-1) == 0, 0).sum(dim=1) / tgt_mask.sum(
                dim=-1).unsqueeze(-1) + 1e-10
            from models.weakly_graph.loss import mmd
            mmd_low_loss = mmd(src_low_pool_feat, tgt_low_pool_feat)
            mmd_high_loss = mmd(src_high_pool_feat, tgt_high_pool_feat)


            # dis_s = self.discriminator(src_pool_feat)
            # dis_t = self.discriminator(tgt_pool_feat)
            # loss_dis_s = torch.mean(torch.sigmoid(dis_s) ** 2)  # self.criterion(dis_s, torch.autograd.Variable(torch.FloatTensor(dis_s.data.size()).fill_(self.source_label)).cuda())
            # loss_dis_t = torch.mean((1 - torch.sigmoid(dis_t)) ** 2)
            # loss_dis = (loss_dis_s + loss_dis_t) / 2
            # loss = loss + 0.001 * loss_dis
            mmd_loss = 10 * (mmd_low_loss)# + mmd_high_loss)
            loss = loss + mmd_loss
            '''



            loss.backward()
            torch.cuda.empty_cache()
            #torch.use_deterministic_algorithms(True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()

            # loss_meter['loss_da_low'].update(mmd_low_loss.item())
            # loss_meter['loss_da_high'].update(mmd_high_loss.item())
            # loss_meter['loss_ent'].update(ent_loss.item())
            loss_meter['loss_ctc'].update(loss_ctc.item())
            loss_meter['loss_ctc_wv'].update(loss_ctc_wv.item())
            loss_meter['loss_sent'].update(loss_sent.item())
            loss_meter['loss_nar'].update(loss_nar.item())
            loss_meter['rec_loss'].update(loss_rec.item())
            # loss_meter['loss_aux'].update(loss_aux.item())
            # loss_meter['loss_wpr'].update(loss_wpr.item())
            loss_meter['loss_quant'].update(loss_quant.item())
            loss_meter['loss_rob'].update(loss_rob.item())
            loss_meter['loss'].update(loss.item())



            if bid % display_n_batches == 0:
                print_log()
                #break
                # if bid % (3 * display_n_batches) == 0:
                #       sc_1, s1 = self.eval(top_n=1, dataloader=self.test_loader, args=self.args)
                #      #sc_1, s1 = self.eval(top_n=1, dataloader=self.test_loader, args=self.args)
                #       sc_5, s5 = self.eval(dataloader=self.test_loader, args=self.args)
                #      if sc_1['IoU@0.7'].avg > 0.13 and sc_5['IoU@0.5'].avg > 0.72 and sc_5['IoU@0.7'].avg > 0.40:
                #          # ave_path = os.path.join(self.model_saved_path, 'tmp-%s.pt' % (str(self.args['train']['lr'])+'-'+s1+'--'+s5))
                #      save_path = os.path.join(self.model_saved_path,
                #                                   'model_seed%d_lr%.5f_topK%d-%.4f_%.4f_%.4f_%.4f_%.4f_%.4f-%d.pt' %
                #                                   (self.args['seed'], self.args['train']['lr'],
                #                                    self.args['train']['topK'],
                #                                    sc_1['IoU@0.1'].avg, sc_1['IoU@0.3'].avg, sc_1['IoU@0.5'].avg,
                #                                    sc_5['IoU@0.1'].avg, sc_5['IoU@0.3'].avg, sc_5['IoU@0.5'].avg,
                #                                    epoch,))
                #      self._save_model(save_path)

        print_log()
        fp.write('=' * 60 + '\n')
        fp.flush()
        fp.close()

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)

            # msg = 'Epoch {}, Batch {},  '.format(epoch, bid)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            logging.info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())
        # loss_meter2 = collections.defaultdict(lambda: AverageMeter())
        if self.args['dataset']['dataset'] == 'ActivityNetGraph':
            num_cands = 153  #213 #153 # 1552 153 1134 134 155 153  143  111  141
            fp = open('append.log', encoding='utf8', mode='a')
        else:
            if self.args['dataset']['dataset'] == 'CharadesSTA':
                num_cands = 528
            else:
                num_cands = 200
            fp = open('append2.log', encoding='utf8', mode='a')
        cnt = 0
        init_memory = None
        # if epoch == 1:
        #     init_memory = self.cluster()
        for bid, batch in enumerate(self.train_loader, 1):
            torch.cuda.empty_cache()
            self.model.train()

            b = len(batch['raw'])
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            net_input['props'] = net_input['props'].expand(len(self.device_ids), -1, -1)
            net_input['props_graph'] = net_input['props_graph'].expand(len(self.device_ids), -1, -1)
            net_input['raw'] = batch['raw']
            if init_memory is not None:
                net_input['init_memory'] = init_memory.cuda()
            #kwargs['is_training'] = True
            kwargs['show'] = False
            kwargs['predict'] = False
            reg_gt = net_input['reg_gt']
            reg_duration = reg_gt[:, 1] - reg_gt[:, 0]
            reg_duration = reg_duration[reg_duration > 16].sum()
            if reg_duration > 1:
                cnt = cnt + reg_duration
            # original branch

            output = self.model(**net_input, get_negative=True, **kwargs)
            torch.cuda.empty_cache()
            # noisy branch
            # net_input['audio_feat'] = net_input['noisy_feat']
            # output_noisy = self.model(**net_input, get_negative=True, **kwargs)

            if kwargs['is_pretrain']:
                loss = output['vae_loss']
                time_meter.update()
                loss_meter['vae_loss'].update(loss.item())
                continue
                # wandb.log({'train_loss': loss})


            torch.cuda.empty_cache()
            #frames_encoded_T = net_input['frames_feat_T'].squeeze(2).cuda()  # [bsz, 32 ,512]
            #index, contrast_idx = net_input['index'].cuda(), net_input['sample_idx'].cuda()
            #loss_kd = self.criterion_kd(sim_frames_encoded, frames_encoded_T, index, contrast_idx)
            #with torch.no_grad():
            #    anchor_id, positive_id, negative_id = self.miner(sim_frames_encoded, labels)

            # loss, nce_loss, dqa_loss, ali_loss, rec_loss = output['loss'], output['nce_loss'], output['dqa_loss'], output['ali_loss'], output['rec_loss']
            # loss_meter['nce_loss'].update(nce_loss.item())
            # loss_meter['dqa_loss'].update(dqa_loss.item())
            # loss_meter['ali_loss'].update(ali_loss.item())
            # loss_meter['rec_loss'].update(rec_loss.item())
            # loss_nce = output['nce_loss']
            # loss_nll = output['nll_loss']
            #loss = loss_nce + 0.1 * loss_nll
            # loss_vq = output['vq_loss']
            # loss_meter['loss_vq'].update(loss_vq.item())
            # loss_sharp = output['sharp_loss']
            # loss_meter['loss_sharp'].update(loss_sharp.item())
            #loss = output['loss']
            # ctc_loss = output['ctc_loss']
            # nce_loss = output['sent_loss']
            # loss = output['loss']
            #
            # loss_meter['ctc_loss'].update(ctc_loss.item())
            # loss_meter['nce_loss'].update(nce_loss.item())

            loss, loss_norm = weakly_supervised_loss_new(output['score'], neg_score1=output['inter_neg']['neg_score'],
                                             props=net_input['props'][0],
                                             log_fp=fp, num_cands=num_cands,
                                             reg_gt=net_input['reg_gt'],
                                             loss_meter=loss_meter, map_gt=net_input['map_gt'], **self.args['train'])
            loss_contrast = output['contrast_fa']
            # ema_loss = output['ema_loss']
            # loss = loss + ema_loss
            # loss_meter['ema_loss'].update(ema_loss.item())

            # dqa_loss = output['dqa_loss']
            # loss = loss + dqa_loss * 0.01
            # loss_meter['dqa_loss'].update(dqa_loss.item())
            from models.weakly_graph.loss import SniCoLoss
            # loss_contrast = SniCoLoss()(output['contrast_pairs'])
            loss_meter['loss_contrast'].update(loss_contrast.item())
            if epoch > 0:
                loss = loss + 0.1 * loss_contrast
            # loss = loss# + 0.1 * loss_sharp#+ loss_nce * 1.0 + 0.1 * loss_nll
            # if epoch > 0:
            #     loss = loss + loss_norm

            # loss, tmp_ori_sm ,inter_loss= weakly_supervised_loss(pos_score=output['score'],
            #                                  clean_score=None,
            #                                  #contrast_pairs=output['contrast_pairs'],
            #                                  loss_rec=None,
            #                                  loss_dcor=output['loss_dcor'],
            #                                  map_iou=self.train_set.map_iou,
            #                                  pos_weight=output['filter_weight'],
            #                                  prob_mat=None,
            #                                  prob_mat_neg=None,
            #                                  start_prob=None,
            #                                  end_prob=None,
            #                                  frame_feat=output['frame_feat'],
            #                                  neg_score1=output['inter_neg']['neg_score'],
            #                                  self_neg_score1=output['inter_neg']['self_neg_score'],
            #                                  neg_score2=output['intra_neg']['neg_score'],
            #                                  neg_weight2=output['intra_neg']['weight'],
            #                                  neg_weight2_graph=output['intra_neg']['weight_graph'],
            #                                  erase_tri_loss=output['erase_tri_loss'],
            #                                  erase_tri_loss_verb=output['erase_tri_loss_verb'],
            #                                  erase_tri_loss_noun=output['erase_tri_loss_noun'],
            #                                  weight_gt=net_input['frame_gt'],
            #                                  map_weight=None,
            #                                  #clip_sim=net_input['map_sim'],
            #                                  #self_sim=net_input['self_sim'],
            #                                  map_gt=net_input['map_gt'],
            #                                  props=net_input['props'][0],
            #                                  log_fp=fp, num_cands=num_cands,
            #                                  reg_gt=net_input['reg_gt'],
            #                                  # contrast_pairs=output['contrast_pairs'],
            #                                  # contrast_fa=output['contrast_fa'],
            #                                  #vqdist=output['vqdist'],
            #                                  loss_meter=loss_meter, **self.args['train'],
            #                                  loss_kd=None,
            #                                  noisy=False
            #                                  )


            #loss = loss + output['sim_loss'] + output['recon_loss'] + 1e-3 * output['dcor_loss']
            #label_score = (net_input['map_sim']).detach().cpu().numpy()
            #frame_score = torch.sigmoid(output['fused_score']).detach().cpu().numpy()
            # attn_weight = output['attn_weight'].detach().cpu().numpy()#.mean(1)
            # word_topk = np.argsort(attn_weight, axis=1)[:, -3:]

            # # 4.add extra pseudo supervise loss
            # if epoch > 3:
            #           loss =  loss + inter_loss
            # loss += output['supervise']['loss'] * 1e-2
            # loss += output['supervise']['loss'] * 1e-1  #* (1 - 0.5 * np.exp(- (self.num_updates + 1) / 100))
            # loss += erase_loss *  np.exp(- (self.num_updates + 1) / 100)


            #torch.use_deterministic_algorithms(False)
            #loss = loss + 0.1 * output['loss_kd']
            torch.cuda.empty_cache()
            loss.backward()
            #torch.use_deterministic_algorithms(True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            # loss_meter['loss_nll'].update(loss_nll.item())
            # loss_meter['loss_nce'].update(loss_nce.item())
            loss_meter['loss'].update(loss.item())



            if bid % display_n_batches == 0:
                print_log()
                # print(output['indices'])

                #break
                # if bid % (3 * display_n_batches) == 0:
                #       sc_1, s1 = self.eval(top_n=1, dataloader=self.test_loader, args=self.args)
                #      #sc_1, s1 = self.eval(top_n=1, dataloader=self.test_loader, args=self.args)
                #       sc_5, s5 = self.eval(dataloader=self.test_loader, args=self.args)
                #      if sc_1['IoU@0.7'].avg > 0.13 and sc_5['IoU@0.5'].avg > 0.72 and sc_5['IoU@0.7'].avg > 0.40:
                #          # ave_path = os.path.join(self.model_saved_path, 'tmp-%s.pt' % (str(self.args['train']['lr'])+'-'+s1+'--'+s5))
                #      save_path = os.path.join(self.model_saved_path,
                #                                   'model_seed%d_lr%.5f_topK%d-%.4f_%.4f_%.4f_%.4f_%.4f_%.4f-%d.pt' %
                #                                   (self.args['seed'], self.args['train']['lr'],
                #                                    self.args['train']['topK'],
                #                                    sc_1['IoU@0.1'].avg, sc_1['IoU@0.3'].avg, sc_1['IoU@0.5'].avg,
                #                                    sc_5['IoU@0.1'].avg, sc_5['IoU@0.3'].avg, sc_5['IoU@0.5'].avg,
                #                                    epoch,))
                #      self._save_model(save_path)

        #print('Finish training autoencoder A')
        print_log()
        fp.write('=' * 60 + '\n')
        fp.flush()
        fp.close()



    def eval(self, top_n=5, thresh=0.45, flag=0, threshold=0, dataloader=None, test_props=None, weight=None,**kwargs):
        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())

        spk_res = {}
        for i in range(0, 110):
            spk_res[i] = []
        cnt = 0
        with torch.no_grad():
            for bid, batch in enumerate(dataloader, 1):
                raw = batch['raw']

                net_input = move_to_cuda(batch['net_input'])
                net_input['props'] = net_input['props'].expand(len(self.device_ids), -1, -1)
                net_input['props_graph'] = net_input['props_graph'].expand(len(self.device_ids), -1, -1)
                net_input['tgt'] = False
                kwargs['is_training'] = False
                kwargs['predict'] = False
                output = self.model(**net_input, get_negative=True, is_pretrain=False, get_noise=False, **kwargs)
                # # pre-training stage
                #
                # # quant_loss = output['quant_loss']
                # nce_loss_, nce_loss_wv_, quant_loss_tf_ = output
                # # print(quant_loss.item())
                # nce_loss = nce_loss + nce_loss_
                # nce_loss_wv = nce_loss_wv + nce_loss_wv_
                # quant_loss_tf = quant_loss_tf + quant_loss_tf_
                # cnt = cnt + 1
                # continue
                # return 0, 0

                durations = np.asarray([i[1] for i in batch['raw']])
                gt = np.asarray([i[2] for i in batch['raw']])
                #loss, prob = bce_rescale_loss(output['score'],  net_input['map_gt'])
                prob = torch.sigmoid(output['score'])

                #metrics_logger['loss'].update(loss.item())
                bsz = prob.size(0)
                prob = np.reshape(prob.cpu().numpy(), [bsz, -1]) # [64, 528]
                idx = np.argmax(prob, -1)
                selected_props = self.test_set.props[idx]  # [bsz, 2]


                if top_n > 1:
                    num_clips = self.num_clips
                    sort_idx = np.argsort(-prob, -1)
                    cand_props = list(self.test_set.props[sort_idx])  # [bsz, cand_props, 2]

                    #valid_prop = valid_prop[sort_idx]
                    top_n_selected_props = [selected_props]
                    top_n_selected_ids = [sort_idx]

                    for _ in range(1, top_n):
                        ptr_props = top_n_selected_props[-1]
                        selected_props = []
                        selected_ids = []
                        for i in range(bsz):
                            p2 = cand_props[i]
                            p1 = np.repeat(np.expand_dims(ptr_props[i], 0),
                                           p2.shape[0], 0)

                            iou = calculate_IoU_batch2((p1[:, 0], p1[:, 1]), (p2[:, 0], p2[:, 1]))
                            keep = iou <= thresh
                            # print(keep.shape, cand_props[i].shape)
                            cand_props[i] = cand_props[i][keep]
                            valid_keep = np.where(keep == True)[0]
                            # print(cand_props[i].shape)
                            # valid_range = len(cand_props[i])
                            # rand_idx = np.random.randint(0, valid_range)
                            selected_props.append(cand_props[i][0])
                            selected_ids.append(sort_idx[i][valid_keep[0]])
                        top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips

                        # print(np.asarray(selected_props).shape, selected_props[0].shape)
                        top_n_selected_props.append(np.asarray(selected_props))
                        top_n_selected_ids.append(np.asarray(selected_ids))

                    top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips

                    res = top_n_metric(top_n_selected_props, gt)
                else:

                    sp = selected_props * durations[:, np.newaxis] / self.num_clips  # [b,2]

                    # for i in range(len(raw)):
                    #     speaker = int(raw[i][-1])
                    #     spk_res[speaker].append(selected_props[i].tolist())
                    res, iou = top_1_metric(sp[:], gt[:])


                for k, v in res.items():
                    metrics_logger[k].update(v, bsz)

        s = ''
        for k, v in metrics_logger.items():
            s += '|' + str(k)[:6]+' ' + str(v.avg)[:6] +' '
            print('| {} {:.4f}'.format(k, v.avg), end=' ')
        print('|')
        # with open("/home/wangye/wangye2/wangye/TIP2021-erase/predict_wocon.json", 'w') as f:
        #     json.dump(time_res, f)
        # with open("/home1/lihaoyuan/wangye/TIP2021-erase/spk_res_contrast", 'w') as f:
        #     json.dump(spk_res, f)
        # nce_loss = nce_loss / cnt
        # nce_loss_wv = nce_loss_wv / cnt
        # quant_loss_tf = quant_loss_tf / cnt
        # print(nce_loss.item())
        # print(nce_loss_wv.item())
        # print(quant_loss_tf.item())
        return metrics_logger, s

    def _build_dataset(self, mode=None):
        def seed_worker(worker_id):
            # worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])

        args = self.args['dataset']
        if self.stage == 'pre':
            data_path = '/home/wangye/wangye/data/LibriSpeech/train-all-960.json' #'/home/wangye/wangye2/wangye/TIP2021-erase/train_corpus_sent.json'
            vocab_path = '/home/wangye/wangye2/wangye/TIP2021-erase/bert_vocab.bin'
            from dataset.librispeech import LibriSpeech
            dataset = LibriSpeech(data_path=args['train_data'], vocab_path=args['vocab_path'], glove_sent_path=args['glove_sent_path'], args=args)
            # # exit(0)
            self.new_train_loader = DataLoader(dataset, batch_size=self.args['train']['batch_size'], shuffle=True, num_workers=8,
                                pin_memory=True, collate_fn=dataset.collate_fn)
        else:

            cls = getattr(da, args['dataset'], None)
            if 'ActivityNetGraph' in self.args['dataset']['dataset']:
                vocab = load_json(args['vocab_path'])
            else:
                vocab = KeyedVectors.load_word2vec_format(args['vocab_path'], binary=True)
            self.vocab = vocab
            self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, dataset=args['dataset'])
            self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, dataset=args['dataset'])
            # self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args) if args['val_data'] else None
            # logging.info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
            batch_size = self.args['train']['batch_size']
            self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                           collate_fn=self.train_set.collate_data, num_workers=8, worker_init_fn=seed_worker, pin_memory=True)
            self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                          collate_fn=self.test_set.collate_data, num_workers=4, pin_memory=False)
            # self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
            #                              collate_fn=self.val_set.collate_data,
            #                              num_workers=4) if args['val_data'] else None



    def _build_model(self):
        model_config = self.args['model']
        import models

        device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        logging.info('GPU: {}'.format(device_ids))
        self.model = getattr(models, model_config['name'], None)(model_config)
        # summary(self.model, (1,20,300))
        # exit()

        #self.model = self.model.cuda()
        # self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.model = self.model.cuda(device_ids[0])
        self.device_ids = device_ids
        # print(self.model)

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        parameters = list(self.model.parameters())
        args = self.args['train']
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        logging.info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        # path = os.path.join(self.args.model_saved_path, name)
        state_dict = torch.load(path)
        # self.num_updates = state_dict['num_updates']
        # self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters, strict=False)
        logging.info('load model from {}.'.format(path))



def entropy(p):
    # print(torch.isnan(p))
    # p_np = p.detach().cpu().numpy()
    p = F.softmax(p, dim=-1)

    out = -torch.mean(torch.sum(p * torch.log(p + 1e-10), 1))
    return out


if __name__ == '__main__':
    a = np.array([1, 6])
    b = np.array([2, 1])
    print(np.maximum(a, b))
    a[a < b] = b[b > a]
    print(a)
