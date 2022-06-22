import os
import torch.nn as nn
import torch.optim
from tqdm import tqdm, trange
import time
from colorama import Fore, Back, Style
import logging
import argparse
import yaml
from easydict import EasyDict
from models import TARN
from dataloader import get_dataloader
import utils
   
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--gpu", type=int, default=0)
    parser.add_argument("-w","--num_workers", type=int, default=3)
    args = parser.parse_args()

    config.gpu = args.gpu
    config.num_workers = args.num_workers

    #append
    config.feature_extractor_type = "Conv4"
    config.n_shot = 5

    model = TARN(config)
    logger_name = config.metric_type
    logger, logfile_name = utils.create_logger(logger_name)
    model.set_logger(logger, logfile_name)
    model.logger.info(config)

    total = sum([param.nelement() for param in model.parameters()])
    print("#### %.2fM"%(total/1e6))
    # model.train_process()
    # model.test_process("weights/47_70.56_0.78.pkl")

if __name__ == "__main__":
    main()