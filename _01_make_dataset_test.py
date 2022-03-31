'''
실행순서 : 01
Created on 2021. 10. 14.

@author: jieun

실행 : python _01_make_dataset_test.py
설명 : data/root/_data/denta 임시 데이터 생성 및 하위 연령별/성별 치아 학습 그룹화 데이터셋 생성
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

class MakeDataset():
    # 초기화
    def __init__(self):
        super().__init__()
        return
    
    
    # argument 추가
    def arg_parser_add(self):
        parser = argparse.ArgumentParser()
        #### MPL ####
        # MPL 학습 대상 이미지 폴더 경로
        parser.add_argument('--data-path', default='./_data/origin', type=str, help='data path')
        # MPL 저장할 데이터셋명
        parser.add_argument('--data-name', default=f'age_cls_{today}_densenet_test', type=str, help='dataset name')
        # MPL 학습 데이터셋 저장경로
        parser.add_argument('--save-path', default='./_data/', type=str, help='dataset save path')
        # MPL 데이터셋 구성 시 train,test 비율
        parser.add_argument('--test-rto', default=0.1, type=float, help='test ratio')
        # MPL 학습 데이터셋 그룹핑 기준 (base : 6세~20세 1세단위 21세~40세 5세단위 41세 이후 7세단위, custom : vgg모델 추론결과에 따라 클래스 그룹화) 
        parser.add_argument('--grp-type', default='base', help='choose grouping type', choices=['base', 'custom'])
        # MPL base 그룹핑 선택시 엑셀 파일 경로 
        parser.add_argument('--grp-path', default='./_config/group_info_base.csv', type=str, help='grouping dataset file path')
        # MPL 그룹데이터 클래스별 수량 지정 (min.100 ~ max.300)
        parser.add_argument('--grp-max', default=100, type=int, help='amount of data per class')
        #### VGG ####
        # VGG 학습 데이터셋 경로
        parser.add_argument('--vgg-data', default='./vgg_test/_data/dataset_211213/', type=str, help='vgg training dataset path')
        # VGG 모델 경로
        parser.add_argument('--vgg-path', default='./VGG16_v2-OCT_Retina_half_dataset_1639381717.7591405.pt', 
                            type=str, help='load vgg model for grouping info')
        # VGG 그룹핑 기준 accuracy
        parser.add_argument('--std-acc', default=10, type=int, help='vgg grouping standard accuracy')
        # VGG 그룹핑 연령 구간 기준
        parser.add_argument('--age', default=10, type=int, help='vgg grouping standard range of age')
        # ---------------------------------------------------------------------------------------------------
        # 로컬 테스트용(은평성모 환경에서는 사용하지 않음) 
        # 샘플 데이터셋 생성 여부(DDH 이미지)
        parser.add_argument('--req-sample', default='n', type=str, help='Use sample image & dataset')
        # 샘플 데이터 경로(DDH 이미지)
        parser.add_argument('--img-path', default=r'D:\nia_dental_files\new\all_ori\image/', type=str, help='img path')
        # ---------------------------------------------------------------------------------------------------
        args = parser.parse_args()
        return args
    
    
    # # 샘플 이미지 및 경로 구성
    # def make_sample(self, args):
    #     # TODO: path 설정, 디렉토리 생성 관련
    #     data_root = args.data_path 
    #     img_root = args.img_path
    #     # data 폴더 하위에 root/_data/denta 폴더까지 생성
    #     dir_root = os.path.join(data_root, 'root')
    #     dir_data = os.path.join(dir_root, '_data')
    #     dir_denta = os.path.join(dir_data, 'age_clss')
    #     path_list = [dir_root, dir_data, dir_denta]
    #     for _,path in enumerate(path_list):
    #         if os.path.exists(path) == False:
    #             os.makedirs(path, exist_ok=True)
    #     # denta 폴더 내 "연령(3자리)_성별" 폴더 생성 (000_F~100_M)
    #     age = ["{0:03}".format(i) for i in range(101)]  # 000 ~ 100
    #     gender = ['M', 'F']                             # Male, Female
    #     for _,num in enumerate(age):
    #         for _,mf in enumerate(gender):
    #             os.makedirs(os.path.join(dir_denta, num+'_'+mf), exist_ok=True)
    #     # DDH 치아 파노라마 이미지 10장씩 폴더에 복사
    #     # 파노라마 원본 이미지 리스트(파일형식 : png)
    #     img_list = [ _ for _ in os.listdir(img_root) if _.endswith('.png')]
    #     # 복사된 이미지가 저장될 폴더의 리스트(000_F ~ 100_M)
    #     dir_list = os.listdir(dir_denta)
    #     # 10개씩 갯수 분할
    #     list_sub = [img_list[i:i+10] for i in range(0, len(img_list), 10)] 
    #     # 각 폴더로 파노라마 이미지 10개씩 복사(대상파일, 대상폴더)
    #     for i,dir in enumerate(dir_list):
    #         for _,img in enumerate(list_sub[i]):
    #             shutil.copy(os.path.join(img_root, img), os.path.join(dir_denta, dir))
    #     return
    
    
    # base group 적용 (0~5세: 제외, 6~20세: 1년단위, 21~40세: 5년단위, 41세 이후: 7년단위)
    def base_grpInfo(self, args):
        # (기존클래스-새로운클래스)로 매핑된 csv 파일 열기
        logger.info(f"    group info file path : {args.grp_path}")  # base 클래스 정보 저장된 파일 경로
        grp_file = os.path.join(args.grp_path)
        col_names = ["ori_class_nm", "grp_class_nm"]
        df = pd.read_csv(grp_file, names=col_names, keep_default_na=False)  # Na -> None
        ori_class_nm = df.ori_class_nm.to_list()    # 기존 클래스 리스트
        grp_class_nm = df.grp_class_nm.to_list()    # 그룹화 클래스 리스트
        class_dict = dict(zip(ori_class_nm, grp_class_nm))
        for i,grp in enumerate(ori_class_nm):
            # 0~5세: 제외
            if class_dict[grp] == '': del class_dict[grp]
        new_class_list = list(class_dict.values())  # 최종 클래스 리스트(190개)
        return class_dict, new_class_list
    
    
    # vggNet 결과에 따라 데이터셋 구성
    def custom_grpInfo(self, args, save_set):
        # load VGG test dataset
        data_dir = args.vgg_data    # data_dir = 'vgg_test_211118/_data/dataset'
        TRAIN = 'train'
        VAL = 'val'
        TEST = 'test'
        class_names, image_datasets, dataloaders, dataset_sizes = load_dataset(data_dir, TRAIN, VAL, TEST)
        test_loader = dataloaders[TEST]
        
        # Load the pretrained model from pytorch
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
        # Freeze training for all layers
        for param in vgg.features.parameters():
            param.require_grad = False
        # Newly created modules have require_grad=True by default
        num_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, len(class_names))])    # Add our layer with 4 outputs
        vgg.classifier = nn.Sequential(*features)   # Replace the model classifier
        vgg.to(args.device) # vgg.requires_grad_(False), vgg.eval()
        # load saved model 
        checkpoint = torch.load(f'{args.vgg_path}', map_location=torch.device('cpu'))
        vgg.load_state_dict(checkpoint)
        # confusion matrix
        nb_classes = len(class_names)
        con_mat = np.zeros((nb_classes, nb_classes))
        # predict test dataset
        class_score = []
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(test_loader):
                inputs = inputs.to(args.device)
                classes = classes.to(args.device)
                outputs = vgg(inputs)
                _, preds = torch.max(outputs, 1)    # 추론이미지별 acc 가장 높은 클래스인덱스
                _, indices = torch.sort(outputs, descending=True)   # 추론이미지별 acc 가장 높은 순서대로 모든 클래스인덱스 나열
                for i, (t,p) in enumerate(zip(classes.view(-1), preds.view(-1))):
                    # 가로 : predicted label, 세로 : GT label
                    con_mat[t.long(), p.long()] += 1
                    percent = torch.nn.functional.softmax(outputs, dim=1)[i] * 100
                    # GT idx & pred idx & score desc
                    tp_score = [(t.item(), idx.item(), percent[idx].item()) for idx in indices[i,]] # class_names[idx]
                    class_score.append(tp_score)
        
        # confusion matrix to dataframe
        df_cm = pd.DataFrame(con_mat, index=class_names, columns=class_names).astype(int)
        # vgg confusion matrix to png
        plt.ioff()
        plt.figure(figsize=(75, 50))
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(save_set, f'vgg_cm.png'), dpi=300)
                    
        # 전체 추론결과에서  오답인데 일정 accuracy 이상인 오답&정답 그룹핑
        gt_pred = []
        for img in class_score:
            for i in img:
                gt, pred, acc = i[0], i[1], i[2]
                #if gt != pred and acc >= args.std_acc and abs(gt - pred) <= args.age:
                if gt != pred and acc >= args.std_acc: 
                    gt_pred.append((gt, pred, acc))
        # std_acc 또는 age 값 조정 필요
        assert len(gt_pred) != 0, 'Adjust std_acc or age value'
        
        #print("gt_pred=", gt_pred)
        
        group = []
        for i in gt_pred:
            gt_nm, pred_nm = class_names[i[0]], class_names[i[1]]
            # 성별분리(다른 성별로 추론된것은 그룹핑 제외..?)
            if gt_nm[-1] == pred_nm[-1] : group.append((gt_nm, pred_nm))
        # gt & pred 튜플 내 순서 관계없이 중복 제거
        uq_group = list(set(map(tuple, map(sorted, group))))
        
        print("++++++++++++++++++++++++++++++++++++++++++")
        print("uq_group=", uq_group)
        
        # 튜플 내 같은값이 존재하면 그룹핑
        s = [set(i) for i in uq_group if i]
        
        def find_intersection(m_list):
            for i,v in enumerate(m_list):
                for j,k in enumerate(m_list[i+1:], i+1):
                    if v & k:
                        s[i] = v.union(m_list.pop(j))
                        return find_intersection(m_list)
            return m_list
        group_info = find_intersection(s)
        
        print("==========================================")
        print("group_info=", group_info)

        # 그룹명 부여
        group_nm = []
        for i,item in enumerate(group_info):
            group_nm.append("GROUP_"+str(i))
        # 기존 클래스 & 그룹 클래스 매핑
        ori_class_nm = os.listdir(args.data_path)
        grp_class_nm = ori_class_nm.copy()
        for i,cls in enumerate(grp_class_nm):
            for j,grp in enumerate(group_info): # list in tuples
                if cls in grp : grp_class_nm[i] = group_nm[j]
        class_dict = dict(zip(ori_class_nm, grp_class_nm))  # dict
        # # 0~5세: 제외 
        # for nm in ori_class_nm:
        #     if int(nm[:3]) in range(6): 
        #         del class_dict[nm]
        new_class_list = list(class_dict.values())  # 최종 클래스 리스트
        # 그룹핑 클래스 정보 csv 저장
        df = pd.DataFrame(list(class_dict.items()), columns=['ori_class_nm', 'grp_class_nm'])
        return class_dict, new_class_list, df
    
    
    # 데이터셋 구성
    def main(self):
        # argument 추가
        args = self.arg_parser_add()
        args.gpu = 0
        args.device = torch.device('cuda', args.gpu)    #cuda:0

        # data/dental-data/train&test 폴더 생성
        save_set = os.path.join(args.save_path, args.data_name)  # 실제 학습용 dataset
        save_train = os.path.join(save_set, 'train')        # train
        save_test = os.path.join(save_set, 'test')          # test
        save_val = os.path.join(save_set, 'val')          # val
        path_list = [args.save_path, save_set, save_train, save_test, save_val]
     
        for _,path in enumerate(path_list):
            if os.path.exists(path) == False:
                os.makedirs(path, exist_ok=True)
        
        # 샘플 이미지 및 롤더 생성(DDH 이미지 사용 시=Y)
        if args.req_sample == 'y':
            self.make_sample(args)

        # custom grouping
        if args.grp_type == 'custom' :
            dict_class, new_class_nm, df_group = self.custom_grpInfo(args, save_set) 
            # vgg group info to csv
            df_group.to_csv(os.path.join(save_set, 'custom_group_info.csv'), header=None, index=False)
        # base grouping
        else :  
            dict_class, new_class_nm = self.base_grpInfo(args)
    
        # 각 폴더에 담긴 이미지 리스트 가져오기
        # dir_list = os.listdir(data_root)    # 폴더(클래스)리스트
        # 폴더리스트에서 0~5세 제외
        dir_list = list(dict_class.keys())
        # (기존 클래스명-변경내역) 매핑된 dict 및 그룹화된 클래스명 리스트
        unique_list = list(dict.fromkeys(new_class_nm)) # 클래스 인덱스 생성 위한 순서유지&중복제거된 리스트
        
        class_idx = []
        class_nm = []
        img_nm = []
        ori_class_nm = []
        class_data_num = dict() # 클래스별 이미지 갯수 dict
        # 이미지명, 클래스명
        for _,dir in enumerate(dir_list):
            try:
                for _,img in enumerate(os.listdir(os.path.join(args.data_path, dir))):
                    img_nm.append(img)
                    ori_class_nm.append(dir)
                    class_nm.append(dir) if dict_class[dir] == '' else class_nm.append(dict_class[dir])
                class_data_num[dir] = len(os.listdir(os.path.join(args.data_path, dir)))
            except Exception as e:
                print("No directory : ", dir) # 클래스 정보 csv에 있지만, 원본 폴더(이미지) 없는 클래스
                pass
        print("class_data_num=",class_data_num)
        # 클래스인덱스
        for _,nm in enumerate(class_nm):    
            class_idx.append(unique_list.index(nm))
        
        # list to dataframe
        df = pd.DataFrame([ _ for _ in list(zip(class_idx, class_nm, ori_class_nm, img_nm))])
        print(df)
        # column name 지정
        df.columns = ['class_idx', 'class_nm', 'ori_class_nm', 'img_nm']       
        # 원본 df 백업용
        df.to_csv(os.path.join(save_set, 'data_org.csv'), index=False)
        
        # 같은 값 그룹명을 가진 class의 갯수
        grp_list = list(dict_class.values())
        cls_per_grp = { i : grp_list.count(i) for i in grp_list }
        
        # 조건 1. 각 클래스별 그룹 구성시 갯수가 균등하게 분배
        # 조건 2. 각 그룹별 데이터의 총량이 args.grp_max를 넘지 않음
        # df_filter = df.groupby('ori_class_nm', group_keys=False).apply(lambda x : print(cls_per_grp[dict_class[x.name]])).reset_index(drop=True)
        #-----------------------------------------------------------------------------------------------------------------------------     
        # 220110 클래스별 최소 200개씩 추출
        # if args.grp_type == 'custom' :
        #     # df = df.groupby('ori_class_nm', group_keys=False).apply(lambda x : print(x.name, int(args.grp_max / cls_per_grp[dict_class[x.name]]))).reset_index(drop=True) 
        #     # 86세 이상 데이터가 모두 100개 미만이어서, args.grp_max>=100으로 설정하면 ValueError 발생
        #     # df = df.groupby('ori_class_nm', group_keys=False).apply(lambda x : x.sample(
        #     #     n=int(class_data_num[x.name] / cls_per_grp[dict_class[x.name]]))).reset_index(drop=True) 
        #     df = df.groupby('ori_class_nm', group_keys=False).apply(lambda x : 
        #             x.sample(n=int(class_data_num[x.name] / cls_per_grp[dict_class[x.name]])) 
        #                 if class_data_num[x.name] < args.grp_max 
        #                 else x.sample(n=int(args.grp_max / cls_per_grp[dict_class[x.name]]))
        #             ).reset_index(drop=True)
        # else:    
        #     # df = df.groupby('ori_class_nm', group_keys=False).apply(lambda x : x.sample(
        #     #     n=int(args.grp_max / cls_per_grp[dict_class[x.name]]))).reset_index(drop=True) 
        #     df = df.groupby('ori_class_nm', group_keys=False).apply(lambda x : 
        #                 x.sample(n=int(class_data_num[x.name] / cls_per_grp[dict_class[x.name]])) 
        #                     if class_data_num[x.name] < args.grp_max 
        #                     else x.sample(n=int(args.grp_max / cls_per_grp[dict_class[x.name]]))
        #                 ).reset_index(drop=True)
        #-----------------------------------------------------------------------------------------------------------------------------  
        # 220125 클래스별 갯수 최대로 추출(개별 클래스 데이터셋과 갯수 유사하게 하기 위함)
        df = df.groupby('ori_class_nm', group_keys=False).apply(lambda x : 
                    x.sample(n=int(class_data_num[x.name])) 
                        if class_data_num[x.name] < args.grp_max 
                        else x.sample(n=int(args.grp_max))
                    ).reset_index(drop=True)
        #-----------------------------------------------------------------------------------------------------------------------------     
        df.to_csv(os.path.join(save_set, 'filtered_df.csv'), index=False)
    
        # 클래스별 train:test 분할
        x = df.copy()
        y = df['class_nm']
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_rto, stratify=y)
        #------------------------------------------------------------------------------------
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
        # # validation 추가 (0.25 * 0.8 = 0.2)
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) 
        #------------------------------------------------------------------------------------
        # train:test:val = 8:1:1
        x_train, x_test = train_test_split(x, test_size=0.2, stratify=y)
        x_val, x_test = train_test_split(x_test, test_size=0.5, random_state=1) 
        #------------------------------------------------------------------------------------
        # 데이터셋 하위에 train,test,원본 csv 저장
        print(len(x_train))
        print(len(x_val))
        print(len(x_test))
        
        print(x_train.head())
        print(x_val.head())
        print(x_test.head())
        
        # labels to csv (idx&label_name)
        label_df = x['class_nm'].unique()
        label_df = pd.Series(label_df)
        label_df.to_csv(os.path.join(save_set, 'label_info.csv'), header=None, index=True)
        # 컬럼 지정
        col_selector = ['img_nm', 'class_idx']
        # train,test
        x_train[col_selector].to_csv(os.path.join(save_set, 'data_label.csv'), header=None, index=False)
        x_test[col_selector].to_csv(os.path.join(save_set, 'data_test.csv'), header=None, index=False)   
        x_val[col_selector].to_csv(os.path.join(save_set, 'data_val.csv'), header=None, index=False)
        #-----------------------------------------------------------------------------------------------------------------------------     
        # 220110 컬럼 추가 추출
        # column name 지정
        x_train.columns = ['class_idx', 'class_nm', 'ori_class_nm', 'img_nm']   
        x_test.columns = ['class_idx', 'class_nm', 'ori_class_nm', 'img_nm'] 
        x_val.columns = ['class_idx', 'class_nm', 'ori_class_nm', 'img_nm']  
        x_train.to_csv(os.path.join(save_set, 'data_label(all).csv'), index=False)
        x_test.to_csv(os.path.join(save_set, 'data_test(all).csv'), index=False) 
        x_val.to_csv(os.path.join(save_set, 'data_val(all).csv'), index=False) 
        #-----------------------------------------------------------------------------------------------------------------------------    
        # train, test 폴더에 이미지 저장
        # 복사되는 대상 이미지 경로 : args.data_path
        # 복사된 이미지 저장경로(test, train폴더) : save_train, save_test
        with tqdm(total=x_train.shape[0], desc="save train img") as pbar:   # train
            for _,row in x_train.iterrows():
                pbar.update(1)
                shutil.copy(os.path.join(args.data_path, row['ori_class_nm'], row['img_nm']), save_train)
        
        with tqdm(total=x_test.shape[0], desc="save test img") as pbar:     # test
            for _,row in x_test.iterrows():
                pbar.update(1)
                shutil.copy(os.path.join(args.data_path, row['ori_class_nm'], row['img_nm']), save_test)
        
        with tqdm(total=x_val.shape[0], desc="save val img") as pbar:     # val
            for _,row in x_val.iterrows():
                pbar.update(1)
                shutil.copy(os.path.join(args.data_path, row['ori_class_nm'], row['img_nm']), save_val)
        return
    
    
if __name__ == "__main__":
    data = MakeDataset()
    data.main()