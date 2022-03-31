import os
import pandas as pd

# D:\xai-workspace\mpl_prj\working\_data\origin
path = r'D:\xai-workspace\mpl_prj\working\_data\origin'
save_path = r'D:\xai-workspace\mpl_prj\working\_data'
dir_list = os.listdir(path)

cnt = []
for i in dir_list:
    dir_path = os.path.join(path, i)
    len_dir = os.listdir(dir_path)
    cnt.append(len(len_dir))

df = pd.DataFrame([ x for x in zip(dir_list, cnt)])
print(len(df))

df.to_csv(os.path.join(save_path, 'target_jpg_20211213.csv'))
print("saved!!")


