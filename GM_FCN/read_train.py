import numpy as np

# c1 = np.loadtxt(indir+'/train/cov1.csv',delimiter=",")
# m1 = np.loadtxt(indir+'/train/mean1.csv',delimiter=",")
# c2 = np.loadtxt(indir+'/train/cov2.csv',delimiter=",")
# m2 = np.loadtxt(indir+'/train/mean2.csv',delimiter=",")
#
# e=np.vstack((c1,m1))
# print(e.shape)
# e=np.vstack((e,c1,m1))
# print(e.shape)
# e=e.reshape(-1,13,12)
# print(e.shape)


#功能：将训练集中的每个实例转化成的cov和mean进行拼接，汇总在一个文件中，便于读取
# train

indir = 'F:/data/12.UWave/3.Cov_Mean'

c1 = np.loadtxt(indir+'/train/cov1.csv',delimiter=",")
m1 = np.loadtxt(indir+'/train/mean1.csv',delimiter=",")

all_cov_mean = np.vstack((c1,m1))
all_cov = np.loadtxt(indir+'/train/cov1.csv',delimiter=",")
for i in range(2,201):
    c = np.loadtxt(indir + '/train/cov'+str(i)+'.csv', delimiter=",")
    m = np.loadtxt(indir + '/train/mean'+str(i)+'.csv', delimiter=",")

    all_cov_mean = np.vstack((all_cov_mean,c, m))

    all_cov = np.vstack((all_cov,c))

np.savetxt(indir + '/train/all_cov_mean.csv', all_cov_mean, fmt='%.3f', delimiter=',')
np.savetxt(indir + '/train/all_cov.csv', all_cov, fmt='%.3f', delimiter=',')
