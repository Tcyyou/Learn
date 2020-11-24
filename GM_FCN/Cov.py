import numpy as np

#运行须知：确定数据集的位置indir，两个for循环对应的上限是跟训练集和测试集实例个数相关（训练集实例个数+1，测试集实例个数+1）
#只将数据集转换为模型参数cov和mean,不处理每个实例的标签

indir = 'F:/data/12.UWave'
#tcy:train->cov,mean  暂时未处理标签文件
for i in range(1,201):#1,2,3...28
    # ‘/Users/zhc/0.DL4TSC/UCR_Time_Series_Classification_Deep_Learning_Baseline-master/data/3.Libras/train_no_class_segmentData’
    train = np.loadtxt(indir+'/1.original/train/train'+str(i)+'.csv',delimiter=",") #shape:(length,dim)
    c = np.cov(np.transpose(train))#shape:(dim,dim)
    m = np.mean(train, axis=0) #shape:(dim)

    np.savetxt(indir+'/3.Cov_Mean/train/cov'+str(i)+'.csv', c, fmt='%.3f', delimiter=',')
    np.savetxt(indir+'/3.Cov_Mean/train/mean' + str(i) + '.csv', m,fmt='%.3f', delimiter=',')
#tcy:test->cov,mean
for i in range(1,4279):
    # ‘/Users/zhc/0.DL4TSC/UCR_Time_Series_Classification_Deep_Learning_Baseline-master/data/3.Libras/test_no_class_segmentData’
    test = np.loadtxt(indir+'/1.original/test/test'+str(i)+'.csv',delimiter=",")#shape:(length,dim)
    c = np.cov(np.transpose(test))#shape:(dim,dim)
    m = np.mean(test, axis=0)#shape:(dim)

    np.savetxt(indir+'/3.Cov_Mean/test/cov'+str(i)+'.csv', c, fmt='%.3f', delimiter=',')
    np.savetxt(indir+'/3.Cov_Mean/test/mean' + str(i) + '.csv', m, fmt='%.3f', delimiter=',')