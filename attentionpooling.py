import tensorflow as tf
from tensorflow.keras.layers import *

class AttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, hdims, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.hdims,
            kernel_initializer=self.kernel_initializer,
            # kernel_regularizer="l2",
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1,
            # kernel_regularizer="l1", # 添加稀疏性
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        w = self.k_dense(inputs)
        w = self.o_dense(w)
        # 处理 mask
        w = w - (1 - mask) * 1e12
        # 权重归一化
        w = tf.math.softmax(w, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(w * x0, axis=1)
        return x, w

class AdditiveAttention1D(tf.keras.layers.Layer):

    def __init__(self, hdims, kernel_initializer="glorot_uniform", **kwargs):
        super(AdditiveAttention1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.hdims,
            kernel_initializer=self.kernel_initializer,
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1,
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x, q = inputs
        q = tf.expand_dims(q, axis=1) # (batch_size, 1, hdims)
        # 计算每个 time steps 权重
        x = self.k_dense(x)
        w = x * q
        w = self.o_dense(w)
        # 处理 mask
        w = w - (1 - mask) * 1e12
        # 权重归一化
        w = tf.math.softmax(w, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(w * x, axis=1)
        return x, w

class MultiHeadAttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, hdims, heads, kernel_initializer="glorot_uniform", **kwargs):
        super(MultiHeadAttentionPooling1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.heads = heads
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        # k_dense可以理解长特征维度的变换，不参与多头相关的操作
        # 因此这里参数不变。当然也可以参与多头操作，但是参数会变多。
        self.k_dense = tf.keras.layers.Dense(
            units=self.hdims,
            kernel_initializer=self.kernel_initializer,
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=self.heads,
            # kernel_regularizer="l1", # 添加稀疏性
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        w = self.k_dense(inputs)
        w = self.o_dense(w)
        # 处理 mask
        w = w - (1 - mask) * 1e12
        # 权重归一化
        # (batch_size, seqlen, heads)
        w = tf.math.softmax(w, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        # （batch_size, seqlen, heads, 1) * (batch_size, seqlen, 1, hdims) 
        # 这里直接对原始输入进行加权平均，因此要考虑维度要一致
        x = tf.reduce_sum(
            tf.expand_dims(w, axis=-1) * tf.expand_dims(x0, axis=2),
            axis=1
        )
        x = tf.reshape(x, (-1, self.heads * self.hdims))
        return x, w

class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        return tf.reduce_max(x, axis=1)
