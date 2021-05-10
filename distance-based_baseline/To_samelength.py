import csv
import numpy as np
from scipy.interpolate import interp1d
import os

# 全局变量:需要根据数据集的训练样本和测试样本的情况填写,路径根据自己的路径设置
#trainset_num = 275
#testset_num = 300
#root= 'E:/baseline/baseline/001--TS-DataSet/ArticularyWordRecognition/'


def transform_to_same_length(x, n_var, max_length):
    # n = x.shape[0]
    n=len(x)
    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)
    # loop through each time series
    for i in range(n):
        # mts = x[i]
        #由于从csv中读取的数据集的shape=（n，l，v），表示n个样本，v维，每一个维度的数据个数为v
        # 而原始代码要求的shape=(n,v,l)
        # 所以重新转置
        mts = np.transpose(np.array(x[i]))
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts
    return ucr_x


def transform_to_npy():
    for path, dirs, files in os.walk(r"E:/baseline/baseline/001--TS-DataSet"):
        for dir in dirs:
            root=os.path.join(path,dir)  #每个数据集的目录
            print(root)
        #每个数据集的长度
            trainset_num=0
            testset_num=0
            for root2, dir2, files2 in os.walk(root+"/1.original/train"):
                trainset_num=len(files2)-1
                break
            for root2, dir2, files2 in os.walk(root+"/1.original/test"):
                testset_num=len(files2)-1
                break

            train_x=read_csv(root + '/1.original/train/train', trainset_num)
            train_y=np.loadtxt(root + '/1.original/train/train_label.csv',delimiter=',')
            test_x = read_csv(root + '/1.original/test/test', testset_num)
            test_y = np.loadtxt(root + '/1.original/test/test_label.csv',delimiter=',')


            out_dir = root + '/2.same_length/'
            n_var = np.array(train_x[0]).shape[1] #40,100,6,此处取变量个数
            max_length = get_func_length(train_x, test_x, func=max)
            min_length = get_func_length(train_x, test_x, func=min)
            print('max', max_length, 'min', min_length)
            # continue
            x_train = transform_to_same_length(train_x, n_var, max_length)
            x_test = transform_to_same_length(test_x, n_var, max_length)

            # save them
            np.save(out_dir + 'train_x.npy', x_train)
            np.save(out_dir + 'train_y.npy', train_y)
            np.save(out_dir + 'test_x.npy', x_test)
            np.save(out_dir + 'test_y.npy', test_y)
            print('trainset:',x_train.shape,train_y.shape)
            print('testset:',x_test.shape,test_y.shape)
            print('Done')
        break


def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = len(x_train)
    for i in range(n):
        func_length = func(func_length, np.array(x_train[i]).shape[0])

    n = len(x_test)
    for i in range(n):
        func_length = func(func_length, np.array(x_test[i]).shape[0])

    return func_length


def read_csv(filepath,sample_num):
    res = []
    for i in range(sample_num):
        with open(filepath + str(i + 1) + '.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            sample = []
            for row in reader:
                sample.append(row)
        res.append(sample)
    return res


def main():
    transform_to_npy()
    # train_x =np.load(root+'2.same_length/train_x.npy')
    # train_y = np.load(root + '2.same_length/train_y.npy')
    # test_x =np.load(root+'2.same_length/test_x.npy')
    # test_y = np.load(root + '2.same_length/test_y.npy')
    # print('trainset:',train_x.shape,train_y.shape)
    # print('testset:',test_x.shape,test_y.shape)

if __name__ == '__main__':
    main()