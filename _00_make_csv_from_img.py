'''
* @fileName : 00_make_csv_from_img.py
* @author : ssum
* @date : 2021-10-07
* @description : image 폴더로부터 csv파일을 생성
* ===========================================================
* DATE AUTHOR NOTE
* -----------------------------------------------------------
* 2021-10-07 ssum-2 최초 생성
'''

import os
import glob
import pandas as pd
import _config.class_list as cg_cls

data_root = 'mpl_prj/working/_data/dental/'
file_name = 'data_label.csv'

class_list = cg_cls.class_list
with open(data_root + file_name, 'w') as fp:
    filelist=glob.glob('{}/**/*.*'.format(data_root), recursive=True)
    
    for ofile in filelist:
        cls_file_spl=ofile.replace('\\', '/').split('/')
        fp.write('{},{}\n'.format(cls_file_spl[-1], cls_file_spl[-2]))
        

# test_file_name = 'data_test.csv'
# with open(data_root + test_file_name, 'w') as fp:
#     for i in range(0, 50):
#         fp.write('cat.{},0\n'.format(i))
#         fp.write('dog.{},1\n'.format(i))
