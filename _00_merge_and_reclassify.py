import os
import glob
import shutil
import argparse
import pandas as pd
import datetime


def add_to_dir_name_filename(origin_path):
    save_path = '{}_all'.format(origin_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    all_filelist=glob.glob('{}/**/*.{}'.format(origin_path, origin_path.split('_')[-1]), recursive=True)

    for filepath in all_filelist:
        filepath=filepath.replace('\\', '/')
        filepath_spl=filepath.split('/')
        savefile='{}/{}__{}'.format(save_path, filepath_spl[-2], filepath_spl[-1])

        if not os.path.exists(savefile):
            shutil.copy(filepath, savefile)

        print('completed copy: {} >>> {}'.format(filepath, savefile))


def reclassify_by_filename(origin_path):
    # save_path='{}_reclassified_211201'.format(origin_path)
    save_path='_data/origin'
    print(origin_path)
    # all_filelist = glob.glob('{}/*.{}'.format(origin_path, origin_path.split('_')[-2]))
    all_filelist = glob.glob('{}/*.jpg'.format(origin_path))
    print(len(all_filelist))
    # 211102 0세 ~ 49세
    # 054_M__R000068533_RGG970102G_2020-09-02_PX_1_cbaa9
    # all_filelist = [o_f for o_f in all_filelist if int(os.path.basename(o_f).split('_')[0]) < 50]
    
    for filepath in all_filelist:
        filepath = filepath.replace('\\', '/')
        filepath_spl = filepath.split('/')
        save_dir='{}/{}'.format(save_path, filepath_spl[-1].split('__')[0])
        filename=filepath_spl[-1].split('__')[-1]
        savefile = '{}/{}'.format(save_dir, filename)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(savefile):
            shutil.copy(filepath, savefile)
        print('completed copy: {} >>> {}'.format(filepath, savefile))

def count_class_by_folder(target_path):
    result_list = []
    
    dirlist=os.listdir(target_path)
    for dirname in dirlist:
        dir_size=os.path.getsize('{}/{}'.format(target_path, dirname))
        file_cnt=len(os.listdir('{}/{}'.format(target_path, dirname)))
        
        result_list.append((dirname, file_cnt, dir_size))    
    df=pd.DataFrame(columns=['dirname', 'count', 'size'], data=result_list)
    now=datetime.datetime.now().strftime('%Y%m%d')
    df.to_csv('{}/aggregated_{}.csv'.format(target_path, now))
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--func', type=str, required=False, default='aggregate')
    parser.add_argument('--path', type=str, required=False, default='./_data/origin')
    args=parser.parse_args()

    # root; C:\Users\DENT\Desktop\선별작업중 target_jpg_all_완료+211130
    # reclassified; ./_data/origin/
    # 추가자료; C:\Users\DENT\Desktop\추가자료 선별작업중\0-50
    if args.func == 'merge':
        add_to_dir_name_filename(args.path)
    elif args.func == 'reclassify':
        reclassify_by_filename(args.path)
    elif args.func == 'aggregate':
        count_class_by_folder(args.path)
