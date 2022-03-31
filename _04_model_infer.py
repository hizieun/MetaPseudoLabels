'''
실행순서 : 04
Created on 2021. 10. 27.

@author: jieun

실행 : python _04_model_infer.py
설명 : 개별 이미지 추론(inference) : csv 파일 클래스별 score 내림차순으로 출력 
'''
import os
import argparse
import numpy as np
import logging
import pandas as pd

import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision import transforms
from skimage import io 
from PIL import Image

from _02_data import DATASET_GETTERS
from models import WideResNet
from utils import model_load_state_dict

logger = logging.getLogger(__name__)



class ModelInfer():
    # 초기화
    def __init__(self):
        super().__init__()
        return
    
    # argument 추가
    def arg_parser_add(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--img-path', type=str, default='./_data/age_cls_20211213/train/R000000011_RGG970102G_2018-09-19_PX_1_fd3ca.jpg', help='test image path')
        parser.add_argument('--data-path', default='./_data/age_cls_20211213', type=str, help='data path')
        parser.add_argument('--save-path', type=str, default='./_result/model_211224130757/infer/', help='test result csv save path')
        parser.add_argument('--weight-path', default='./_result/model_211224130757/checkpoint/age_cls_20211224_best.pth.tar', type=str, help='model path')
        parser.add_argument('--num-classes', default=55, type=int, help='number of classes')
        parser.add_argument('--dataset', default='custom', type=str,
                            choices=['cifar10', 'cifar100','custom'], help='dataset name')
        parser.add_argument('--resize', default=32, type=int, help='resize image')
        args = parser.parse_args()
        return args
    
    # 클래스 리스트 생성
    def csv2list(self, args):
        logger.info(f"    label path : {args.data_path}") # 클래스 정보 저장된 파일 경로
        label_file = os.path.join(args.data_path, 'label_info.csv')
        col_names = ["class_idx", "class_nm"]
        df = pd.read_csv(label_file, names=col_names)
        class_idx = df.class_idx.to_list()
        class_nm = df.class_nm.to_list()
        class_dict = dict(zip(class_idx, class_nm))
        return class_nm
    
    # 이미지 추론
    def infer_img(self, args, model):
        logger.info(f"    image path : {args.img_path}")  # 추론대상 이미지
        normal_mean = (0.5, 0.5, 0.5)
        normal_std = (0.5, 0.5, 0.5)
        
        # preprocess input image
        preprocess = transforms.Compose([
                        # transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        # 2021.12.28 변경 시작
                        transforms.Resize(size=(args.resize, args.resize)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(size=args.resize,
                                              padding=int(args.resize*0.125),
                                              padding_mode='reflect'),
                        # 2021.12.28 변경 끝
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=normal_mean,
                            std=normal_std
                    )])
        # img
        ori_img = Image.open(args.img_path)
        ori_img = ori_img.convert('RGB')
        # img to tensor
        img_t = preprocess(ori_img)
        # unsqueeze(0) : 0인 차원 생성
        x = torch.unsqueeze(img_t, 0)
        output = model(x)
        # labels list
        labels = self.csv2list(args)
        # accuracy
        acc = F.softmax(output, dim=1)[0]*100
        
        # label
#         _, indices = torch.sort(output)
        _, indices = torch.sort(output, descending=True)        
        res = [(labels[idx], acc[idx].item()) for idx in indices[0]]
        print(res)
        return res
    
    # 클래스별 score csv 파일 추출
    def extrc_csv(self, args, res):
        logger.info(f"    save path : {args.save_path}")
        res_csv = 'infer_result.csv'
        save_file = os.path.join(args.save_path, res_csv)
        col_names = ['class', 'accuracy']
        data = np.array(res)
        df = pd.DataFrame(data, columns = col_names)
        df.to_csv(save_file, index=False)
        return
    
    # 메인
    def main(self):
        
        args = self.arg_parser_add()
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # 추론 대상 이미지 경로
        logger.info(f"    weight path : {args.weight_path}")
        # dataset별 depth, widen_factor 지정
        if args.dataset == 'cifar10':
            depth, widen_factor = 28, 2
        elif args.dataset == 'cifar100':
            depth, widen_factor = 28, 8
        elif args.dataset == 'custom':
            depth, widen_factor = 28, 8  # TODO: WRN WideResNet을 위한 factor  depth는 layer의 수, widen은 filter 관련 ?? 
        # student_dropout = 0으로 고정
        model = WideResNet(num_classes=args.num_classes,
                            depth=depth,
                            widen_factor=widen_factor,
                            dropout=0,
                            dense_dropout=0)
        checkpoint = torch.load(args.weight_path, map_location=torch.device('cpu'))
        model_load_state_dict(model, checkpoint['student_state_dict'])
        model.requires_grad_(False)
        model.eval()
        # 이미지 추론
        res = self.infer_img(args, model)
        # 결과 df 받아서 csv 추출 
        self.extrc_csv(args, res)
        return

# 메인함수 호출
if __name__ == '__main__':
    infer = ModelInfer()
    infer.main()