from sktime.utils.load_data import load_from_tsfile_to_dataframe
import os
import sktime
import numpy as np
import pandas as pd
import csv

# 全局变量需要修改
DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "F:/BasicMotions/")
datasetname ='BasicMotions'


train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, datasetname+"_TRAIN.ts"))
test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, datasetname+"_TEST.ts"))


#处理训练数据集，一个样本转换为一个csv文件
list_train_x=train_x.values.tolist()

for i in range(len(list_train_x)):
    list2=list_train_x[i]
    list3=zip(*list2)
    with open(DATA_PATH+'1.original/train/train'+str(i+1)+'.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in list3:
            writer.writerow(row)
# 将标签统一用0，1，2...来进行表示（标签对应的序号可能跟标签出现的先后没有关系）
list_train_y = pd.Categorical(train_y).codes #数据类型为：numpy.ndarray
np.savetxt(DATA_PATH+'1.original/train/train_label.csv',list_train_y,fmt='%d',delimiter=',')



#处理测试数据集，一个样本转换为一个csv文件
list_test_x=test_x.values.tolist()

for i in range(len(list_test_x)):
    list2=list_test_x[i]
    list3=zip(*list2)
    with open(DATA_PATH+'1.original/test/test'+str(i+1)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in list3:
            writer.writerow(row)

list_test_y = pd.Categorical(test_y).codes #数据类型为：numpy.ndarray
np.savetxt(DATA_PATH+'1.original/test/test_label.csv',list_test_y,fmt='%d',delimiter=',')
