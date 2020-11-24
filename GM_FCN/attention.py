import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply,  Concatenate, Conv2D, Add, Activation, Lambda

############################### 通道注意力机制 ###############################
# 使用 1×1卷积替换全连接层。
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(in_planes // ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(1e-4),
                                   use_bias=True)
        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(1e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = layers.Reshape((1, 1, avg.shape[1]))(avg)  # shape (None, 1, 1 feature)
        max = layers.Reshape((1, 1, max.shape[1]))(max)  # shape (None, 1, 1 feature)
        avg_out = self.conv2(tf.nn.relu(self.conv1(avg)))
        max_out = self.conv2(tf.nn.relu(self.conv1(max)))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        return out*inputs

    def get_config(self):
        config = {}
        base_config = super(ChannelAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# #  定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
# def regularized_padded_conv(*args, **kwargs):
#     return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False,
#                          kernel_initializer='he_normal',
#                          kernel_regularizer=regularizers.l2(5e-4))

############################### 空间注意力机制 ###############################
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = layers.Conv2D(1, kernel_size=kernel_size, strides=1,padding='same', use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4), activation=tf.nn.sigmoid)

    def call(self, inputs):
        # inputs.shape=(None,x,y,z)
        avg_out = tf.reduce_mean(inputs, axis=3,keepdims=True)
        max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
        # out.shape=(None,x,y,2)
        out = tf.concat([avg_out, max_out], axis=3)  # 创建一个维度,拼接到一起concat。
        out = self.conv1(out)
        return out*inputs

    def get_config(self):
        config = {}
        base_config = super(SpatialAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)

    se_feature = multiply([input_feature, se_feature])
    return se_feature
