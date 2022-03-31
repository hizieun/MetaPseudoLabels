'''
실행순서 : 01
Created on 2021. 12. 06.

@author: jieun

실행 : python _01_extract_data.py
설명 : 009_M ~ 020_M 중복 제외한 데이터 추가 추출
'''
import pandas as pd
import os
import shutil
import glob
import numpy as np


def load_csv_files(patient_path, prescription_path):
    patient_df = pd.read_csv(patient_path, encoding='utf-8')
    patient_df = patient_df[['연구번호', '생년월', '성별']]
    patient_df.rename(columns={'연구번호': 'ID', '생년월': 'Birth', '성별': 'Gender'}, inplace=True)
    # print(patient_df.head())

    prescription_df = pd.read_csv(prescription_path, encoding='utf-8')
    prescription_df = prescription_df[['ID', 'Order Code', 'Order Date', 'Instance Count']]
    # print(prescription_df.head())
    print('prescription_df={}, patient_df={}'.format(len(prescription_df), len(patient_df)))
    return patient_df, prescription_df


def merge_df_on_patientId(patient_df, prescription_df):
    merge_df = pd.merge(prescription_df, patient_df, on='ID')
    merge_df['Order Date'] = pd.to_datetime(merge_df['Order Date'].astype('str'))
    merge_df['Birth'] = pd.to_datetime(merge_df['Birth'].astype('str'))
    merge_df=merge_df.dropna(axis=0)
    print('merge_df={}'.format(len(merge_df)))
    return merge_df


# 나이 계산
def age(order_date, birth):
    age = order_date.year - birth.year - ((order_date.month, order_date.day) < (birth.month, birth.day))
    str_age = str(age).zfill(3)
    return str_age


# 모든 원본 파일에 대한 df
def get_all_df(ori_path):
    all_df = pd.DataFrame() # 모든 파일 리스트
    
    # 원본 데이터 리스트
    ori_dir_all = os.listdir(ori_path)  # origin 폴더 내 전체
    print(len(ori_dir_all))
    ori_dir = []
    for i in ori_dir_all:
        if os.path.isdir(os.path.join(ori_path, i)) == True:
            ori_dir.append(i)
    print(len(ori_dir))
    for i,one in enumerate(ori_dir):
        dir_path = os.path.join(ori_path, one)
        dir_in = os.listdir(dir_path)
        for j,name in enumerate(dir_in):
            path = os.path.join(ori_path, one, name)
            try:
                if name[-4:] == '.csv':
                    list_df = pd.read_csv(path, encoding='utf-8')
                    # print(list_df.columns)
                    all_df = all_df.append(list_df, ignore_index=True)
            except UnicodeDecodeError as e:
                print(path)
                pass
        
    # origin\R000000002_RGG970102G_2019-04-17\201904171284872392\1\RGG970102G_2019-04-17_PX_1_83f54.dcm
    # glob로 dcm 파일 경로 찾기
    dcm_target = './datas/ep_panorama/AgeEval/**/*.dcm'
    dcm_list = glob.glob(dcm_target, recursive=True)
    all_df['Path'] = None
    for i,row in all_df.iterrows():
        for _,dcm in enumerate(dcm_list):
            if all_df.loc[i, 'File'] == dcm.split(os.path.sep)[4]:
                all_df.loc[i, 'Path'] = dcm
    all_df['Order Date'] = pd.to_datetime(all_df['Order Date'].astype('str'))
    all_df['dir_name'] = all_df.apply(lambda x: '{}_{}_{}'.format(
                        x['ID'], x['Order Code'], x['Order Date'].strftime('%Y-%m-%d')), axis=1)
    return all_df


# 새로 저장할 파일이 기존 추출된 리스트에 있는지 중복 체크하여 저장할 이미지 정보만 담긴 df 생성
def check_redundancy(save_root, sample_path, all_df):
    saved_dir = os.listdir(save_root)
    sample_df = pd.read_csv(sample_path, encoding='utf-8') # 기존 저장된 리스트
    saved_dir_list = sample_df['dir_name'].to_list()
    # print(len(saved_dir_list)) # 25646
    checked_df = all_df.copy()
    # print(len(checked_df))
    for i,row in checked_df.iterrows():
        if row['dir_name'] in saved_dir_list: # ID_OrderCode_OrderDate
           checked_df = checked_df.drop(i)
    # print(len(checked_df))
    ## TODO : saved_dir에 동일한 파일명 존재하는지 체크!!
    
    return checked_df


# dcm 파일 복사(처방일에 따른 연령정보 필요)
def copy_to_tmp(checked_df, saved_path):
    try:
        for i,row in checked_df.iterrows():
            # 추가 추출대상 : 009_M ~ 020_M
            if int(row['save_dir'][:3]) in range(9,21):
                ori_path = row['Path']
                save_path = os.path.join(saved_path, row['save_dir'])
                # 저장할 폴더 없으면 생성
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                shutil.copy(ori_path, save_path)
        
                result = '>>> completed copy to target_path'
            else:
                result = '>>> no target file to copy'
                
    except Exception as e:
        result='!!! Failed copy to target path because of {}'.format(e)
    return result


if __name__ == '__main__':
    '''
         추출 전 원본경로 : /origin/ -> 아래 구조 참고    
        상위폴더 (origin_path)
            ㄴ 연구번호_처방코드_처방일자(ex. R000000002_RGG970102G_2019-04-17) = ID_Order Code_Order Date
                ㄴ 뭔지 모를 일련번호
                    ㄴ 환자 정보 및 dicom 이미지 정보.csv
                    ㄴ dicom폴더
                        ㄴ ~~~.dcm
                    
         추출 후 저장경로 : /tmp/target_jpg/000_성별(save_dir)/*.dcm
    '''
    patient_path='./data/은평 파노라마_COHT_이상화.csv'    # 환자정보
    prescription_path='./data/AgeEval.csv'    # 파일목록    
    
    patient_df, prescription_df = load_csv_files(patient_path, prescription_path)
    merge_df = merge_df_on_patientId(patient_df, prescription_df)
    
    # Age 추가
    merge_df['age'] = merge_df.apply(lambda x: age(x['Order Date'], x['Birth']), axis=1)
    merge_df['save_dir'] = merge_df.apply(lambda x: '{}_{}'.format(x['age'], x['Gender']), axis=1)
    # merge_df.to_csv('./data/py-workspace/data/merge_df.csv', encoding='utf-8', sep=',')
    
    # dcm 원본 데이터 목록
    ori_path = './datas/ep_panorama/AgeEval'
    all_df = get_all_df(ori_path)
    # all_df.to_csv('./data/py-workspace/data/all_df.csv', encoding='utf-8', sep=',')
    
    # ID,Order Date 기준으로 merge_df -> all_df save_dir 정보 가져오기
    new_all_df = pd.merge(left=all_df, right=merge_df, how="inner", left_on=['ID', 'Order Date'], right_on=['ID', 'Order Date'])
    # new_all_df.to_csv('./data/py-workspace/data/new_all_df.csv', encoding='utf-8', sep=',')
    
    # 기존 추출된 데이터 리스트(sampling_set.csv)에 중복인지 체크
    save_root = './target_jpg_20211208/'
    sample_path = './data/sampling_set.csv' # 기존에 추출된 각 클래스별 200건의 데이터 정보
    checked_df = check_redundancy(save_root, sample_path, new_all_df)
    # checked_df.to_csv('./data/py-workspace/data/checked_df.csv', encoding='utf-8', sep=',')
    
    # 009_M ~ 020_M dcm 파일 복사
    result = copy_to_tmp(checked_df, save_root)
    print(result)