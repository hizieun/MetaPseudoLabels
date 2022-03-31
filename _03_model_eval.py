'''
실행순서 : 03
Created on 2021. 10. 27.

@author: jieun

설치 : pip install seaborn
실행 : python _03_model_eval.py
설명 : 모델검증(evaluate) : main.py evaluate 분리, confusion matrix 생성
'''
import argparse
import logging
import math
import os
import random
import time
# confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import torch
from tqdm import tqdm
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

# from data import DATASET_GETTERS
from _02_data import DATASET_GETTERS
from _04_model_infer import ModelInfer
from models import WideResNet, ModelEMA
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict)


class ModelEval():
    def __init__(self):
        super().__init__()
        return
     
    def set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # argument 추가
    def arg_parser_add(self): 
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-labeled', type=int, default=2750, help='number of labeled data') # class개수 * 클래스별 라벨링처리할 데이터건수(10~50)
        parser.add_argument('--save-path', type=str, default='./_result/model_211228222155/infer/')
        parser.add_argument('--weight-path', default='./_result/model_211228222155/checkpoint/age_cls_20211224_best.pth.tar'
                            , type=str, help='model path')
        parser.add_argument('--data-path', default='./_data/age_cls_20211224/', type=str, help='dataset path')
        parser.add_argument('--num-classes', default=55, type=int, help='number of classes')
        parser.add_argument('--batch-size', default=8, type=int, help='train batch size')
        # uda unsupervised data augmentation 
        parser.add_argument('--workers', default=1, type=int, help='number of workers')
        parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
        parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
        parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument('--resize', default=32, type=int, help='resize image')
        parser.add_argument('--csv-train-filename', default='data_label.csv', type=str, help='csv filname')
        parser.add_argument('--csv-test-filename', default='data_test.csv', type=str, help='csv filname')
        parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
        parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
        parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
        parser.add_argument('--dataset', default='custom', type=str,
                            choices=['cifar10', 'cifar100','custom'], help='dataset type')
        args = parser.parse_args()
        return args
     
    def evaluate(self, args, test_loader, model, criterion):
        batch_time = AverageMeter()  # 새로 설정 저장 및 평균
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        model.eval()  # eval 모드
        test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
        all_preds = torch.tensor([]) # confusion matrix
        con_mat = np.zeros((args.num_classes, args.num_classes))
        class_score = []

        with torch.no_grad():
            end = time.time()
            for step, (images, targets) in enumerate(test_iter):
                data_time.update(time.time() - end)
                batch_size = images.shape[0]
                images = images.to(args.device)
                targets = targets.to(args.device)
                # with amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
                soft_pseudo_label = torch.softmax(outputs.detach(), dim=-1)
                # confusion matrix
                all_preds = all_preds.to(args.device)
                all_preds = torch.cat(
                    (all_preds, outputs), dim=0
                )
                _, preds = torch.max(outputs, 1)    # 추론이미지별 acc 가장 높은 클래스인덱스
                _, indices = torch.sort(outputs, descending=True)   # 추론이미지별 acc 가장 높은 순서대로 모든 클래스인덱스 나열
                for i, (t,p) in enumerate(zip(targets.view(-1), preds.view(-1))):
                    # 가로 : predicted label, 세로 : GT label
                    con_mat[t.long(), p.long()] += 1
                    percent = torch.nn.functional.softmax(outputs, dim=1)[i] * 100
                    # GT idx & pred idx & score desc
                    tp_score = [(t.item(), idx.item(), percent[idx].item()) for idx in indices[i,]] # class_names[idx]
                    class_score.append(tp_score)
        
                acc1, acc5 = accuracy(outputs, targets, (1, 2))  # minibatch에서 top 1, top 2를 추출
                losses.update(loss.item(), batch_size)
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                batch_time.update(time.time() - end)
                end = time.time()
                test_iter.set_description(
                    f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                    f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")  # eval mini batch별로 출력 
            test_iter.close()
            # make confusion matrix
        self.make_cm(args, test_loader, all_preds, con_mat)
        return losses.avg, top1.avg, top5.avg

    # make confusion matrix
    def make_cm(self, args, test_loader, all_preds, con_mat):
        labels = test_loader.__dict__['dataset'].__dict__['targets']
        labels = torch.tensor(labels)
        labels = labels.to(args.device)
        preds_correct = all_preds.argmax(dim=1).eq(labels).sum().item()
        print('total correct:', preds_correct)
        print('accuracy:', preds_correct / len(labels) * 100)
        #**
        stacked = torch.stack(
            (
                labels, all_preds.argmax(dim=1)
            )
            ,dim=1
        )
        # class_name 함수 호출
        class_list = ModelInfer()
        class_name = class_list.csv2list(args)
        cmt = torch.zeros(len(class_name), len(class_name), dtype=torch.int64)
        for p in stacked:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl]+1
        cm = confusion_matrix(labels.cpu().numpy(), all_preds.argmax(dim=1).cpu().numpy()) # numpy array
        # confusion matrix (percent)
        plt.figure(figsize=(75, 50))
        class_nm = np.asarray(class_name)
        save_cm = sns.heatmap(cm/np.sum(cm), annot=True, 
                    fmt='.2%', cmap='Blues', xticklabels=class_name, yticklabels=class_name)
        save_cm.set_title('confusion matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')
        save_cm.get_figure().savefig(os.path.join(args.save_path, 'mpl_cm(percent).png'))
        # confusion matrix to DF
        df_cm = pd.DataFrame(con_mat, index=class_name, columns=class_name).astype(int)
        plt.ioff()
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')
        plt.savefig(os.path.join(args.save_path, f'mpl_cm.png'), dpi=300)
        return
        
    def main(self):
        # argument 추가
        args = self.arg_parser_add()
        
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        if args.local_rank != -1:  # pytorch 분산 처리, default -1이 아니면 
            args.gpu = args.local_rank
            torch.distributed.init_process_group(backend='nccl')  # 분산처리를 위한 backend를 nvidia collectiva communication library를 사용
            args.world_size = torch.distributed.get_world_size()  # 분산처리에 참여하는 프로세스의 수 = world size 
        else:
            args.gpu = 0  # 단독작업 
            args.world_size = 1
        
        args.device = torch.device('cuda', args.gpu)  #cuda:0
        
        # 재현을 위하여 난수의 seed값을 고정
        if args.seed is not None:
            self.set_seed(args)   
        
        # dataset별 depth, widen_factor 지정
        if args.dataset == 'cifar10':
            depth, widen_factor = 28, 2
        elif args.dataset == 'cifar100':
            depth, widen_factor = 28, 8 
        # 수정
        elif args.dataset == 'custom':
            depth, widen_factor = 28, 8  # TODO: WRN WideResNet을 위한 factor  depth는 layer의 수, widen은 filter 관련 ?? 
        # elif args.dataset == 'custom':
        #         depth, widen_factor = 34, 32
        
        # DATASET_GETTERS 함수에대한 point를 가지고 있는 dict
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args) 
        print(args.batch_size) 
        test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)
        # dense_dropout -> student_dropout = 0으로 고정
        student_model = WideResNet(num_classes=args.num_classes,
                                   depth=depth,
                                   widen_factor=widen_factor,
                                   dropout=0,
                                   dense_dropout=0)
        
        student_model.to(args.device) # 모델을 device에 할당 
        # f'{args.name}_finetune_last.pth.tar'
        checkpoint = torch.load(os.path.join(args.weight_path), map_location=torch.device('cpu'))
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
        student_model.requires_grad_(False)
        student_model.eval()
        avg_student_model = None
        
        if args.ema > 0:  # exponential moving average를 사용할 때 
            # student model을 이용하여 ema 방식으로 average student model을 생성 
            avg_student_model = ModelEMA(student_model, args.ema)
        
        criterion = create_loss_fn(args)  # crossentropy와 smoothcrossentropy중 선택
        # evaluate 함수 실행 (evaluate 하는 경우, teacher 불필요, unlabel 불필요, label 불필요) 
        self.evaluate(args, test_loader, student_model, criterion)
        return

if __name__ == '__main__':
    eval = ModelEval()
    eval.main()