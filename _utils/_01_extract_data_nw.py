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


def check_duplicate_by_df(save_root, merge_df, sample_df):
    # saved_dir = os.listdir(save_root)

    saved_dir_list = sample_df['dir_name'].to_list()
    print(len(saved_dir_list)) # 25646
    checked_df = merge_df.copy()
    # print(len(checked_df))
    for i,row in merge_df.iterrows():
        if row['dir_name'] in saved_dir_list: # ID_OrderCode_OrderDate
           checked_df = checked_df.drop(i)
    print(len(checked_df))
    ## TODO : saved_dir에 동일한 파일명 존재하는지 체크!!
    
    return checked_df

def move_to_target_path(checked_df, origin_path, save_path):
    try:
        # 연구번호_처방코드_처방날짜
        # R000000002_RGG970102G_2019-04-17
        
        for i, row in checked_df.iterrows():
            print('df index = {}'.format(i))
            ori_one_path = os.path.join(origin_path, row['dir_name'])

            origin_file_list=glob.glob('{}/**/*.dcm'.format(ori_one_path), recursive=True)

            if origin_file_list != []:
                # 한 폴더 하위에 한개의 이미지 파일만 있으므로 index 고정
                origin_file_path=origin_file_list[0]
                image_file = origin_file_path.split('/')[-1]

               # 저장경로와 저장할 파일의 전체 경로 정의
                save_image_file='{}_{}'.format(row['ID'], image_file)
                target_path=os.path.join(save_path, row['save_dir'])
                save_file_path=os.path.join(target_path, save_image_file)

                if not os.path.isdir(target_path):
                    os.makedirs(target_path, exist_ok=True)

                if not os.path.exists(save_file_path):
                    shutil.copy(origin_file_path, save_file_path)
                print('completed copy: {} >>> {}\n'.format(origin_file_path, save_file_path))
            else:
                continue
        result = '>>> completed move to target_path'
    except Exception as e:
        result='!!! Failed move to target path because of {}'.format(e)
    return result

  
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
    
    #merge_df = merge_df.astype({'age':'str'})
    #print(merge_df.dtypes)
    print(merge_df.head())
    filter_list = [ str(i).zfill(3) for i in range(9,21) ]
    print(filter_list)
    
    df_filter=merge_df['age'].isin(filter_list)
    merge_df=merge_df[df_filter]
    print(merge_df.head())
    
    merge_df['dir_name'] = merge_df.apply(lambda x: '{}_{}_{}'.format(x['ID'], x['Order Code'], x['Order Date'].strftime('%Y-%m-%d')), axis=1)    
    merge_df['save_dir'] = merge_df.apply(lambda x: '{}_{}'.format(x['age'], x['Gender']), axis=1)
    # merge_df.to_csv('./data/py-workspace/data/merge_df.csv', encoding='utf-8', sep=',')
    
    sample_path = './data/sampling_set.csv' # 기존에 추출된 각 클래스별 200건의 데이터 정보
    sample_df = pd.read_csv(sample_path, encoding='utf-8')
    #df_filter=sample_df['age'].isin(l)
    df_filter=sample_df['age'].isin(list(range(9,21)))
    #sample_df=sample_df[merge_df]
    sample_df=sample_df[df_filter]
    print(sample_df.head())
    print(len(sample_df))

    ori_path = './datas/ep_panorama/AgeEval'
    save_root = './target_jpg_20211208/'
    
    # dcm 원본 데이터 목록
    #all_df = get_all_df(ori_path)
    # all_df.to_csv('./data/py-workspace/data/all_df.csv', encoding='utf-8', sep=',')
    
    # ID,Order Date 기준으로 merge_df -> all_df save_dir 정보 가져오기
    #new_all_df = pd.merge(left=all_df, right=merge_df, how="inner", left_on=['ID', 'Order Date'], right_on=['ID', 'Order Date'])
    # new_all_df.to_csv('./data/py-workspace/data/new_all_df.csv', encoding='utf-8', sep=',')
    
    # 기존 추출된 데이터 리스트(sampling_set.csv)에 중복인지 체크
    
    
    #checked_df = check_redundancy(save_root, sample_path, new_all_df)
    # checked_df.to_csv('./data/py-workspace/data/checked_df.csv', encoding='utf-8', sep=',')
    checked_df = check_duplicate_by_df(save_root, merge_df, sample_df)
    print("checked_df : ", len(checked_df))
    
    # 009_M ~ 020_M dcm 파일 복사
    #result = copy_to_tmp(checked_df, save_root)
    result = move_to_target_path(checked_df, ori_path, save_root)
    print(result)