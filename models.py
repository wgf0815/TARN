import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from dataloader import get_dataloader
import utils
from tqdm import tqdm
from colorama import Fore, Back, Style
from itertools import chain
from feature_extractor import Conv4
from tarn import BasicBlock,ResNet,resRN,resTCA,RN,TCA
from collections import OrderedDict
 
class TARN(nn.Module):
    def __init__(self, config):
        super(TARN, self).__init__()
        self.config = config
        # feature_dim = config.feature_dim

        is_fm = True if self.config.metric_type not in ["RN", "DCA"] else False
        
        if config.feature_extractor_type == "Conv4":
            self.feature_extractor = Conv4()
            feature_dim = 64
        elif config.feature_extractor_type == "ResNet":
            self.feature_extractor = ResNet(BasicBlock, [1, 1, 1, 1], drop_rate=0.0, max_pool=True)
            feature_dim = 640
        
        if config.metric_type == "RN":
            self.metric_network = RN()
        elif config.metric_type == "TCA":
            self.metric_network = TCA(feature_dim)
        elif config.metric_type == "resRN":
            self.metric_network = resRN(BasicBlock,[1,1],drop_rate=0.0,max_pool=True)
        elif config.metric_type == "resTCA":
            self.metric_network = resTCA(feature_dim,BasicBlock,[1,1],drop_rate=0.0,max_pool=True)
        else:
            print("Wrong metric_type: %s" % config.metric_type)
            exit()
        
        if config.feature_extractor_type == "Conv4":
            self.optimizer = torch.optim.Adam(self.parameters(), lr = config.lr, weight_decay=0)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[40000,50000,65000,80000], gamma=0.1)
        elif config.finetune == True:
            param_list = [{"params": self.metric_network.parameters()},
                        {"params": self.feature_extractor.parameters()}]
            self.optimizer = torch.optim.SGD(param_list, lr = config.lr, weight_decay=0.0005, momentum = 0.9)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[40000,50000,65000,80000], gamma=0.1)
        else:
            param_list = [{"params": self.metric_network.parameters()}]
            self.optimizer = torch.optim.SGD(param_list, lr = config.lr, weight_decay=0.0005, momentum = 0.9)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[40000,50000,65000,80000], gamma=0.1)

        self.cuda(self.config.gpu)
        self.apply(weights_init)

        if config.feature_extractor_type == "ResNet" and config.pretrained == True:
            if config.dataset_root == "../../miniImagenet":
                state_dict = torch.load('pretrained_weight/mini-ImageNet/model.pth')
            else:
                state_dict = torch.load('pretrained_weight/tiered-ImageNet/model.pth')
            new_state = OrderedDict()
            for k,v in state_dict.items():
                if k[:18] == 'feature_extractor.':
                    new_state[k[18:]]=v
            self.feature_extractor.load_state_dict(new_state,strict=True)
        self.best_model_file = None
        self.no_boost_num = 0
        self.milestones = []
        self.best_acc = 0
        self.h = 0

    def set_logger(self, logger, logfile_name):
        self.logger = logger
        self.logfile_name = logfile_name

    def forward(self, sample_images, query_images):
        sample_images = sample_images[0].cuda(self.config.gpu)
        query_images = query_images[0].cuda(self.config.gpu)
        n_way = self.config.n_way
        sample_features = self.feature_extractor(sample_images)
        query_features = self.feature_extractor(query_images)
        feature_shape = sample_features.shape[1:]
        sample_features = sample_features.view(n_way, -1, *feature_shape).mean(1)

        relations = self.metric_network(sample_features, query_features)

        return relations

    def train_loop(self, epoch, train_loader):
        avg_loss = 0
        for episode, (sample_images, query_images) in enumerate(train_loader):
            
            relations = self.forward(sample_images, query_images)

            y = torch.from_numpy(np.repeat(range(self.config.n_way), self.config.n_query_train)).cuda(self.config.gpu)
            one_hot_labels = torch.zeros((len(y), self.config.n_way)).cuda(self.config.gpu).scatter_(1, y.unsqueeze(1), 1)
            M = torch.eye(self.config.n_way).cuda(self.config.gpu)
            Msq = one_hot_labels.permute(1,0)
            target = torch.LongTensor([i//self.config.n_query_train for i in range(self.config.n_query_train*self.config.n_way)]).cuda(self.config.gpu)
            Mss = torch.LongTensor([i//self.config.n_way for i in range(self.config.n_way)]).cuda(self.config.gpu)
            if self.config.metric_type in ["RN","TCA"]:
                loss_type = nn.MSELoss()
            else:
                loss_type = nn.NLLLoss()
            if self.config.metric_type == "RN":
                loss = (loss_type(relations[0], one_hot_labels) + loss_type(relations[1], one_hot_labels))/2
            elif self.config.metric_type == "TCA":
                loss = (loss_type(relations[0], one_hot_labels) + loss_type(relations[1], one_hot_labels) + 
                        loss_type(relations[2],M) + loss_type(relations[3],Msq))/2 
            elif self.config.metric_type == "resRN":
                loss = (loss_type(relations[0],target) + loss_type(relations[1],target))/2
            elif self.config.metric_type == "resTCA":
                loss = (loss_type(relations[0],target) + loss_type(relations[1],target) + 
                        loss_type(relations[2],Mss) + loss_type(relations[3],target))/2
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.feature_extractor_type == "Conv4":
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

            last_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            self.scheduler.step(epoch*self.config.val_freq + episode)
            if self.optimizer.param_groups[0]["lr"] != last_lr:
                print(Fore.YELLOW, end="")
                self.logger.info("*** Learning rate decayed ***")
                self.logger.info("*** Best model reloaded ***")
                self.logger.info("Best accuracy = {:4.2f}+-{:4.2f}".format(self.best_acc, self.h))
                print(Style.RESET_ALL, end="")
                self.load_state_dict(torch.load(self.best_model_file))

            avg_loss = avg_loss + loss.item()

            if (episode+1) % self.config.display_freq == 0:
                self.logger.info("Lr {:.5g} | Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
                                self.optimizer.param_groups[0]["lr"], epoch, episode+1, 
                                len(train_loader), avg_loss / self.config.display_freq))
                avg_loss = 0
 
    def val_loop(self, epoch, val_loader):
        is_better = False
        for i in range(3):
            acc, h = self.test_loop(val_loader)
            if acc > self.best_acc:
                is_better = True
                self.best_acc = np.round(acc, 4)
                self.h = np.round(h, 4)
                print(Fore.YELLOW, end="")
                self.logger.info("Val Acc = {:4.2f}+-{:4.2f}".format(acc, h))
                print(Style.RESET_ALL, end="")
            else:
                self.logger.info("Val Acc = {:4.2f}+-{:4.2f}".format(acc, h))

        if is_better:
            self.best_model_file = os.path.join("weights", "{:d}_{:4.2f}_{:4.2f}.pkl".format(
                                                epoch, self.best_acc, self.h))
            torch.save(self.state_dict(), self.best_model_file)
            print(Fore.YELLOW, end="")
            self.logger.info("Saving networks for epoch: {:d}".format(epoch))
            print(Style.RESET_ALL, end="")

    def test_loop(self, test_loader):
        accuracies = []
        with torch.no_grad():
            for episode, (sample_images, query_images) in tqdm(enumerate(test_loader)):
                relations = self.forward(sample_images, query_images)
                relation = (relations[0] + relations[1]) / 2
                predicted_results = relation.data.topk(k=1, dim=1)[1].squeeze().cpu().numpy()
                ground_truth = np.repeat(range(self.config.n_way), self.config.n_query_test)
                correct_num = np.sum(predicted_results == ground_truth)
                query_num = len(predicted_results)
                accuracies.append(correct_num / query_num * 100)
        acc, h = utils.mean_confidence_interval(accuracies)

        return acc, h

    def post_process(self):
        self.logger.info("Best accuracy = {:4.2f}+-{:4.2f}".format(self.best_acc, self.h))
        new_logfile_name = self.logfile_name[:-4] + "_" + str(np.round(self.best_acc, 2)) + "_" + str(np.round(self.h, 2)) + ".log"
        os.rename(self.logfile_name, new_logfile_name)
        self.logger.info("Rename logfile name.")

    def train_process(self):
        train_loader = get_dataloader(self.config, mode="train", n_way=self.config.n_way, n_shot=self.config.n_shot, n_query=self.config.n_query_train)
        val_loader = get_dataloader(self.config, mode="val", n_way=self.config.n_way, n_shot=self.config.n_shot, n_query=self.config.n_query_test)        
        
        if self.config.feature_extractor_type == "ResNet":
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        for epoch in range(50):
            try:
                if (self.config.feature_extractor_type=="ResNet" and self.config.finetune==True and epoch >= 5):
                    for param in self.feature_extractor.parameters():
                        param.requires_grad = True  
                self.train_loop(epoch, train_loader)
                self.val_loop(epoch, val_loader)
            except KeyboardInterrupt:
                if self.best_model_file == None:
                    os.remove(self.logfile_name)
                else:
                    print(Fore.YELLOW, end="") 
                    self.logger.info("KeyboardInterrupt!")
                    self.post_process()
                    print(Style.RESET_ALL, end="")
                exit()

    def test_process(self, best_model_file=None):
        if best_model_file is not None:
            self.best_model_file = best_model_file
        self.load_state_dict(torch.load(self.best_model_file))
        test_loader = get_dataloader(self.config, mode="test", n_way=self.config.n_way, n_shot=self.config.n_shot, n_query=self.config.n_query_test)
        avg_acc = 0
        avg_h = 0
        best_acc = 0
        best_h = 0
        test_num = 10
        for epoch in range(test_num):
            acc, h = self.test_loop(test_loader)
            self.logger.info("Test Acc = {:4.2f}+-{:4.2f}".format(acc, h))
            if acc > best_acc:
                best_acc = acc
                best_h = h
            avg_acc += acc
            avg_h += h
        avg_acc /= test_num
        avg_h /= test_num
        self.logger.info("Best Test Acc = {:4.2f}+-{:4.2f}".format(best_acc, best_h))
        self.logger.info("Avg Test Acc = {:4.2f}+-{:4.2f}".format(avg_acc, avg_h))

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        if m.out_channels%18 != 0:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        else:
            m.weight.data.zero_()
            print("Zero DCN.")
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()




