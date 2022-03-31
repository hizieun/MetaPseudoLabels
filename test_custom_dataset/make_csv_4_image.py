'''
Created on 2021. 8. 19.

@author: tobew
'''

data_root=r'../data/dogs-vs-cats/'
file_name='data_label.csv'
with open(data_root+file_name,'w') as fp :
# with open('data_label.csv','w') as fp :
    for i in range(50,12500,):
        fp.write('cat.{},0\n'.format(i))
        fp.write('dog.{},1\n'.format(i))
        
        
test_file_name='data_test.csv'        
with open(data_root+test_file_name,'w') as fp :
# with open('data_label.csv','w') as fp :
    for i in range(0,50):
        fp.write('cat.{},0\n'.format(i))
        fp.write('dog.{},1\n'.format(i))

        
        