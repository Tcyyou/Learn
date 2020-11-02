import csv
import numpy as np
import math
import os
import time


root ='E:/BasicMotions/'
trainset_num=40
testset_num=40


def distanceDTW(signal_1, signal_2):
    globalDTW = np.zeros((len(signal_1) + 1, len(signal_2) + 1))
    globalDTW[1:, 0] = math.inf
    globalDTW[0, 1:] = math.inf

    globalDTW[0, 0] = 0
    for i in range(1, len(signal_1) + 1):
        for j in range(1, len(signal_2) + 1):
            # 这里的距离度量标准如下：（应该是1NN-DTW-D）
            # np.sum(np.square(np.array(signal_1[i - 1]).astype(float) - np.array(signal_2[j - 1]).astype(float))) 两个一维数组的差的平方之和
            # 后期如果遇到相关论文的明确标准再进行修改
            globalDTW[i, j] = np.sum(np.square(np.array(signal_1[i - 1]).astype(float) - np.array(signal_2[j - 1]).astype(float))) + min(globalDTW[i - 1, j],globalDTW[i, j - 1],globalDTW[i - 1, j - 1])
    return np.sqrt(globalDTW[len(signal_1), len(signal_2)])


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


def classification(train_y, train_x, test_y, test_x):
    if os.path.exists(root+'1NN-DTW-D.txt'):
        os.remove(root+'1NN-DTW-D.txt')

    file_output = open(root+'1NN-DTW-D.txt', 'a+')

    idx = 1
    accuracy = 0
    start = time.time()
    for m in range(len(test_x)):
        min_distance = math.inf
        predict = 0
        for k in range(len(train_x)):

            dist = distanceDTW(test_x[m], train_x[k])
            # Save the min distance
            if dist < min_distance:
                min_distance = dist
                predict = train_y[k]

        file_output.write(
            'Number of series: ' + str(idx) + '	Classification: ' + str(int(predict)) + '	True Class: ' + str(
                int(test_y[m])) + os.linesep)

        if predict == test_y[m]:
            accuracy = accuracy + 1
        print('Processing serie: %d/%d' %(idx,testset_num))
        idx = idx + 1

    end = time.time()
    duration = end - start

    accuracy = accuracy / len(test_x)
    print('accuracy: ', accuracy)
    print('Global Time(seg): ', round(duration, 4))

    file_output.write("Accuracy: " + str(accuracy) + os.linesep)
    file_output.write("Global Time(seg): " + str(round(duration, 4)) + os.linesep)
    file_output.close()


if __name__ == '__main__':
    print('Reading files...')
    train_x=read_csv(root + '1.original/train/train', trainset_num)
    train_y=np.loadtxt(root + '1.original/train/train_label.csv',delimiter=',')
    test_x = read_csv(root + '1.original/test/test', testset_num)
    test_y = np.loadtxt(root + '1.original/test/test_label.csv',delimiter=',')
    print('Done')
    classification(train_y, train_x, test_y, test_x)
