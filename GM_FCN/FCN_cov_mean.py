from __future__ import print_function
from tensorflow import keras
import numpy as np
import pandas as pd

# from GM_FCN.attention import se_block,ChannelAttention,SpatialAttention

import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
  
nb_epochs = 500

# flist  = ['1.JapaneseVowels',
#           '3.Libras',
#           '4.AUSLAN',
#           '5.CharacterTrajectories',
#           '6.ArabicDigits',
#           '7.ECG',
#           '8.Wafer',
#           '9.CMUsubject16',
#           '10.KickvsPunch',
#           '11.NetFlow',
#           '12.UWave',
#           '13.WalkvsRun'
#           'PEMS']

#运行须知：设置flag的值（0则输入cov,1则输入cov_mean）
#设置全局变量dim,使用的数据集维度多少就填多少
#修改数据集路径（也可一次循环所有数据集）

#   输入时，分cov和cov_mean；标号为'all_cov.csv','all_cov_mean.csv'
flag = 0;
#   dim 应该与数据集对应
dim = 22

flist  = ['4.AUSLAN']
batch_size = 128

for each in flist:
    fname = each
    file_name = 'F:/data/'+ each +'/3.Cov_Mean/'

    #   flag = 0;输入为cov
    if(flag == 0):
        x_train = np.loadtxt(file_name + '/train/all_cov.csv', delimiter=',')
        #将cov重塑为FCN输入的格式(None, 62, 62)
        x_train = x_train.reshape(-1, dim,dim)
        y_train = np.loadtxt(file_name + '/train/label.csv', delimiter=',')

        x_test = np.loadtxt(file_name + '/test/all_cov.csv', delimiter=',')
        x_test = x_test.reshape(-1, dim,dim)
        y_test = np.loadtxt(file_name + '/test/label.csv', delimiter=',')

        print('...................')
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print('...................')

        nb_classes = len(np.unique(y_test))
        # batch_size = min(int(x_train.shape[0]/10), 16)

        y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
        y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
        #原始代码
        # Y_train = keras.utils.to_categorical(y_train - 1, nb_classes)
        # Y_test = keras.utils.to_categorical(y_test - 1, nb_classes)
        Y_train = keras.utils.to_categorical(y_train, nb_classes)
        Y_test = keras.utils.to_categorical(y_test, nb_classes)

        x_train_mean = x_train.mean()
        x_train_std = x_train.std()

        x_train = (x_train - x_train_mean) / (x_train_std)
        x_test = (x_test - x_train_mean) / (x_train_std)

        # x_train = x_train.reshape(x_train.shape + (1,1,))
        x_train = x_train.reshape(x_train.shape + (1,))
        # x_test = x_test.reshape(x_test.shape + (1,1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        x = keras.layers.Input(x_train.shape[1:])
        #    drop_out = Dropout(0.2)(x)

        conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)
        #    drop_out = Dropout(0.2)(conv1)
        conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        #    drop_out = Dropout(0.2)(conv2)
        conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)

        # conv3 = ChannelAttention(128)(conv3)
        # conv3 = SpatialAttention()(conv3)
        # or
        # conv3=se_block(conv3)

        conv3 = keras.layers.Activation('relu')(conv3)
        full = keras.layers.GlobalAveragePooling2D()(conv3)
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)

        model = keras.models.Model(inputs=x, outputs=out)

        model.summary()

        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                      patience=50, min_lr=0.0001)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_name + 'Cov_best_model.hdf5', monitor='loss',
                                                           save_best_only=True)
        history = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                            verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr,model_checkpoint])

        # Print the testing results which has the lowest training loss.

        #   保存模型时，分cov和cov_mean；标号为'FCN_1_500.h5','FCN_2_500.h5'
        model.save(file_name + 'FCN_1_500.h5')
        log = pd.DataFrame(history.history)
        #   保存模型时，分cov和cov_mean；标号为'FCN_history_1_500.h5','FCN_history_2_500.h5'
        log.to_csv(file_name + 'FCN_history_1_500.csv')
        print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_accuracy'])
        print(log.loc[log['val_accuracy'].idxmax]['loss'], log.loc[log['val_accuracy'].idxmax]['val_accuracy'])

    #   flag ！= 0;输入为all_cov_mean.csv
    else:

        #分if-else是因为all_cov和all_cov_mean维度不同
        x_train = np.loadtxt(file_name + '/train/all_cov_mean.csv', delimiter=',')
        x_train = x_train.reshape(-1, dim+1, dim)
        y_train = np.loadtxt(file_name + '/train/label.csv', delimiter=',')

        x_test = np.loadtxt(file_name + '/test/all_cov_mean.csv', delimiter=',')
        x_test = x_test.reshape(-1, dim+1, dim)
        y_test = np.loadtxt(file_name + '/test/label.csv', delimiter=',')

        print('...................')
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print('...................')

        nb_classes = len(np.unique(y_test))
        # batch_size = min(int(x_train.shape[0]/10), 16)

        y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
        y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
        #原始代码
        # Y_train = keras.utils.to_categorical(y_train - 1, nb_classes)
        # Y_test = keras.utils.to_categorical(y_test - 1, nb_classes)
        Y_train = keras.utils.to_categorical(y_train, nb_classes)
        Y_test = keras.utils.to_categorical(y_test, nb_classes)

        x_train_mean = x_train.mean()
        x_train_std = x_train.std()

        x_train = (x_train - x_train_mean) / (x_train_std)
        x_test = (x_test - x_train_mean) / (x_train_std)

        # x_train = x_train.reshape(x_train.shape + (1,1,))
        x_train = x_train.reshape(x_train.shape + (1,))
        # x_test = x_test.reshape(x_test.shape + (1,1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        x = keras.layers.Input(x_train.shape[1:])
        #    drop_out = Dropout(0.2)(x)

        conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)
        #    drop_out = Dropout(0.2)(conv1)
        conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        #    drop_out = Dropout(0.2)(conv2)
        conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)

        # conv3 = ChannelAttention(128)(conv3) * conv3
        # conv3 = SpatialAttention()(conv3) * conv3

        full = keras.layers.GlobalAveragePooling2D()(conv3)
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)

        model = keras.models.Model(inputs=x, outputs=out)

        model.summary()

        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                      patience=50, min_lr=0.0001)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_name + 'Cov_Mean_best_model.hdf5', monitor='loss',
                                                           save_best_only=True)
        history = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                            verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr,model_checkpoint ])

        # Print the testing results which has the lowest training loss.

        #   保存模型时，分cov和cov_mean；标号为'FCN_1_500.h5','FCN_2_500.h5'
        model.save(file_name + 'FCN_2_500.h5')
        log = pd.DataFrame(history.history)
        #   保存模型时，分cov和cov_mean；标号为'FCN_history_1_500.h5','FCN_history_2_500.h5'
        log.to_csv(file_name + 'FCN_history_2_500.csv')
        print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_accuracy'])
        print(log.loc[log['val_accuracy'].idxmax]['loss'], log.loc[log['val_accuracy'].idxmax]['val_accuracy'])





############## Get CAM ################
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# get_last_conv = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-3].output])
# last_conv = get_last_conv([x_test[:100], 1])[0]
#
# get_softmax = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output])
# softmax = get_softmax(([x_test[:100], 1]))[0]
# softmax_weight = model.get_weights()[-2]
# CAM = np.dot(last_conv, softmax_weight)
#
#
# # pp = PdfPages('CAM.pdf')
# for k in range(20):
#     CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
#     c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
#     plt.figure(figsize=(13, 7));
#     plt.plot(x_test[k].squeeze());
#     plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r', c=c[k, :, :, int(y_test[k])].squeeze(), s=100);
#     plt.title(
#         'True label:' + str(y_test[k]) + '   likelihood of label ' + str(y_test[k]) + ': ' + str(softmax[k][int(y_test[k])]))
#     plt.colorbar();
#     # plt.show()
#     # pp.savefig()
#
# # pp.close()