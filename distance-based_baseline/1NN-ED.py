import csv
import numpy as np
import math
import os
import time


root ='E:/BasicMotions/'
trainset_num=40
testset_num=40


def distanceED(signal_1, signal_2):
    # 这里的距离度量标准如下：（应该是1NN-ED）
    # 转化
    signal_1 = np.array(signal_1).astype(float)
    signal_2 = np.array(signal_2).astype(float)
    distance = 0
    min_len = min(signal_1.shape[0], signal_2.shape[0])
    max_len = max(signal_1.shape[0], signal_2.shape[0])
    for i in range(0, min_len):
        # 差的平方和
        distance += np.sqrt(np.sum(np.square(signal_1[i] - signal_2[i])))
    if signal_1.shape[0] > signal_2.shape[0]:
        for j in range(min_len,max_len):
            distance += np.sqrt(np.sum(np.square(signal_1[j])))
    elif signal_1.shape[0]<signal_2.shape[0]:
        for j in range(min_len,max_len):
            distance += np.sqrt(np.sum(np.square(signal_2[j])))
    return distance

# def distanceED(signal_1, signal_2):
#     # 这里的距离度量标准如下：（应该是1NN-ED）
#     # 转化
#     signal_1=np.array(signal_1).astype(float)
#     signal_2=np.array(signal_2).astype(float)
#     distance=0
#     # 维度遍历
#     for k in range(0,signal_1.shape[1]):
#         dim_k=0
#         for i in range(0, min(signal_1.shape[0] ,signal_2.shape[0])):
#             # 差的平方和
#             dim_k += np.square(signal_1[i,k]-signal_2[i,k])
#         if signal_1.shape[0]>signal_2.shape[0]:
#             for j in range(min(signal_1.shape[0], signal_2.shape[0]),max(signal_1.shape[0], signal_2.shape[0])):
#                 dim_k+=np.square(signal_1[j,k])
#         if signal_1.shape[0]<signal_2.shape[0]:
#             for j in range(min(signal_1.shape[0], signal_2.shape[0]),max(signal_1.shape[0], signal_2.shape[0])):
#                 dim_k+=np.square(signal_2[j,k])
#         distance+=np.sqrt(dim_k)
#
#     return distance


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
    if os.path.exists(root+'1NN-ED.txt'):
        os.remove(root+'1NN-ED.txt')

    file_output = open(root+'1NN-ED.txt', 'a+')

    idx = 1
    accuracy = 0
    start = time.time()
    for m in range(len(test_x)):
        min_distance = math.inf
        predict = 0
        for k in range(len(train_x)):

            dist = distanceED(test_x[m], train_x[k])
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
