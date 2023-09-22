import json
import time

import numpy as np
import torch


def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)

def load_json_line(filename):
    d = []
    with open(filename, encoding='utf8') as fr:
        for jsonstr in fr.readlines():
            jsonstr = json.loads(jsonstr)
            d.append(jsonstr)
    return d

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


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


def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    if union[1] - union[0] < -1e-5:
        return 0
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    return iou if iou >= 0.0 else 0.0


def calculate_IoU_batch1(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch1((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch1((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result, iou


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n


def signal_by_db(x1, x2, snr, handle_method='append'):
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    if l1 != l2:
        if handle_method == 'cut':
            ll = min(l1, l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            ll = max(l1, l2)
            #print(ll)
            if l1 < ll:
                x2 = x2[:l1]
                #x1 = np.append(x1, x1[:ll-l1])
            if l2 < ll:
                for i in range(int(l1/l2)+5):
                    x2 = np.append(x2, x2[:ll])

                ll2 = min(x1.shape[0], x2.shape[0])
                x1 = x1[:ll2]
                x2 = x2[:ll2]

    from numpy.linalg import norm
    x2 = x2 / (norm(x2) + 1e-10) * norm(x1) / (10.0 ** (0.05 * snr))
    mix = x1 + x2

    return mix, x2

if __name__ == '__main__':
    a = json.load(open('data/activity_net.v1-3.min.json'))['database']['JMpwIWxoB5s']
    print(a['url'])
