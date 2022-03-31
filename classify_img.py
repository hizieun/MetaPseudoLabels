'''
Created on 2022. 1. 13.

@author: jieun

실행 : python classify_img.py
설명 : 데이터셋 클래스별 폴더 분류
python classify_img.py --data-path ./_data/
'''
import os
import shutil
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime

# vgg model
import torch
import torch.nn as nn
import numpy as np
import copy
import seaborn as sns
import time
import matplotlib.pyplot as plt
from vgg_test._02_main import load_dataset

logger = logging.getLogger(__name__)
today = datetime.today().strftime("%Y%m%d")

class Classify():
    # 초기화
    def __init__(self):
        super().__init__()
        return
    
    
    # argument 추가
    def arg_parser_add(self):
        parser = argparse.ArgumentParser()
        #### MPL ####
        # MPL 학습 대상 이미지 폴더 경로
        parser.add_argument('--data-path', default='./_data/age_cls_20211224_backup', type=str, help='dataset path')
        # 데이터 경로(파노라마 이미지)
        # parser.add_argument('--img-path', default=r'./_data/age_cls_20211224/train', type=str, help='ori img path')
        # ---------------------------------------------------------------------------------------------------
        args = parser.parse_args()
        return args
    
    
    # 데이터셋 구성
    def main(self):
        # argument 추가
        args = self.arg_parser_add()

        # ori : './_data/age_cls_20211224'
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'test')
        train_save = os.path.join(args.data_path, 'train_all')
        test_save = os.path.join(args.data_path, 'test_all')  
        path_list = [train_save, test_save]
     
        for _,path in enumerate(path_list):
            if os.path.exists(path) == False:
                os.makedirs(path, exist_ok=True)
        
        train_df = pd.read_csv(os.path.join(args.data_path, 'data_label.csv'), names=['img_nm', 'class_idx'])
        test_df = pd.read_csv(os.path.join(args.data_path, 'data_test.csv'), names=['img_nm', 'class_idx'])
        label_df = pd.read_csv(os.path.join(args.data_path, 'label_info.csv'), names=['class_idx', 'class_nm'])
        
        print("train size:", len(train_df))
        print("test size:", len(test_df))
        print("label cnt:", len(label_df))
        
        merge_train = pd.merge(left=train_df, right=label_df, how="inner", on="class_idx")
        merge_test = pd.merge(left=test_df, right=label_df, how="inner", on="class_idx")
        
        print(merge_train.head())
        print(merge_test.head())
        print("merge_train size:", len(merge_train))
        print("merge_test size:", len(merge_test))
        
        with tqdm(total=merge_train.shape[0], desc="save train img per class") as pbar:   # train
            for _,row in merge_train.iterrows():
                save_dir = os.path.join(train_save, row['class_nm'])
                if os.path.exists(save_dir) == False:
                    os.makedirs(save_dir, exist_ok=True)
                shutil.copy(os.path.join(train_dir, row['img_nm']), save_dir) # 복사대상, 저장경로
                pbar.update(1)
                
        with tqdm(total=merge_test.shape[0], desc="save test img per class") as pbar:   # test
            for _,row in merge_test.iterrows():
                save_dir = os.path.join(test_save, row['class_nm'])
                if os.path.exists(save_dir) == False:
                    os.makedirs(save_dir, exist_ok=True)
                shutil.copy(os.path.join(test_dir, row['img_nm']), save_dir) # 복사대상, 저장경로
                pbar.update(1)
       
        return
    
    
if __name__ == "__main__":
    data = Classify()
    data.main()