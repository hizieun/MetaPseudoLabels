'''
Created on 2021. 8. 19.

@author: tobew
pip install pandas 
pip install scikit-image 
'''

import os 
import pandas as pd 
import torch 
from torch.utils.data import Dataset 
from skimage import io 
from PIL import Image 

class CustomDataset(Dataset):
    def __init__(self,csv_file, root_dir,train=True, 
                 transform=None,
                 target_transform=None   ):
        self.data=pd.read_csv(os.path.join(root_dir,csv_file))
        if train: 
            self.root_dir=os.path.join(root_dir,'train')
        else:
            self.root_dir=os.path.join(root_dir,'test')
                     
        self.transform = transform 
        self.train=train 
        self.target_transform = target_transform 
#         # annotation에서 data를 생성할 것 
#         self.data =[]
#         if train:
        self.targets =list(self.data.iloc[:,1])
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.root_dir, str(self.data.iloc[index,0]))
        # print("_02_custom_dataset.py>line39>img_path=",img_path)
        try:
#             image= io.imread(img_path+'.jpeg') # 변경
            # ---- 변경 코드 시작-----------
            image = io.imread(img_path, pilmode='RGB')
            # ---- 변경 코드 끝-----------
            target=torch.tensor(int(self.data.iloc[index,1]))
    #         img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            image = Image.fromarray(image)

            if self.transform:
                image= self.transform(image)

            if self.target_transform is not None:
                target = self.target_transform(target)
        except Exception as e:
            print(e)
        return (image,target)