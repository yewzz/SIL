import argparse, os

from utils import load_json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default='config/activitynet/config_pretrain.json')  # activitynet  charades
    parser.add_argument('--g', type=str, default='0')
    # parser.add_argument('--b', type=int, default=64)
    return parser.parse_args()


def main(args):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    # b = args.b
    args = load_json(args.config_path)
    # args['train']['batch_size'] = b
    print(args)
    seeds = [0,13,20,25]#, 3, 10, 12, 16] # [12, 5, 8]


    for seed in seeds:
        for x in range(1, 2):
            for y in range(1, 2):
                print(x)
                print(y)
                torch.cuda.empty_cache()
                args['seed'] = seed
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)

                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
                logging.info('base seed {}'.format(seed))
                #logging.info('base learn rate {}'.format(lrs[lr]))
                torch.cuda.empty_cache()

                runner = MainRunner(args)
                runner.train()


if __name__ == '__main__':


    import matplotlib.pyplot as plt

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    main(args)




