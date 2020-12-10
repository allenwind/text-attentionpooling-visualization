import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from attentionpooling import AttentionPooling1D
from dataset import SimpleTokenizer, find_best_maxlen
from dataset import load_lcqmc

# 来自Transformer的激活函数，效果略有提升
def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

class PositionEmbedding(tf.keras.layers.Layer):
    """可学习的位置Embedding"""

    def __init__(self, maxlen, output_dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=output_dim
        )

    def call(self, inputs):
        # maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        return self.embedding(positions)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

X1, X2, y, classes = load_lcqmc()
X1_train = X1[:-1000]
X2_train = X2[:-1000]
y_train = y[:-1000]

X1_test = X1[-1000:]
X2_test = X2[-1000:]
y_test = y[-1000:]

num_classes = len(classes)
tokenizer = SimpleTokenizer()
tokenizer.fit(X1 + X2)
X1_train = tokenizer.transform(X1_train)
X2_train = tokenizer.transform(X2_train)

maxlen = 48
hdims = 128
epochs = 20

X1_train = sequence.pad_sequences(
    X1_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0
)
X2_train = sequence.pad_sequences(
    X2_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0
)
y_train = tf.keras.utils.to_categorical(y_train)

num_words = len(tokenizer)
embedding_dims = 128

x1_input = Input(shape=(maxlen,))
x2_input = Input(shape=(maxlen,))
# 计算全局mask
x1_mask = Lambda(lambda x: tf.not_equal(x, 0))(x1_input)
x2_mask = Lambda(lambda x: tf.not_equal(x, 0))(x2_input)

embedding = Embedding(
    num_words,
    embedding_dims,
    embeddings_initializer="glorot_normal",
    input_length=maxlen
)
# 加上position embedding后，val acc:90%+
posembedding = PositionEmbedding(maxlen, embedding_dims)
layernom = LayerNormalization()

x1 = embedding(x1_input) + posembedding(x1_input)
x1 = layernom(x1)
x1 = Dropout(0.1)(x1)

x2 = embedding(x2_input) + posembedding(x2_input)
x2 = layernom(x2)
x2 = Dropout(0.1)(x2)

conv1 = Conv1D(filters=hdims, kernel_size=2, padding="same", activation=gelu)
conv2 = Conv1D(filters=hdims, kernel_size=3, padding="same", activation=gelu)
conv3 = Conv1D(filters=hdims, kernel_size=4, padding="same", activation=gelu)

x1 = conv1(x1) + x1
x1 = conv2(x1) + x1
x1 = conv3(x1) + x1

x2 = conv1(x2) + x2
x2 = conv2(x2) + x2
x2 = conv3(x2) + x2

pool = AttentionPooling1D(hdims)
x1, w1 = pool(x1, mask=x1_mask)
x2, w2 = pool(x2, mask=x2_mask)

# x*y
x3 = Multiply()([x1, x2])
# |x-y|
x4 = Lambda(lambda x: tf.abs(x[0] - x[1]))([x1, x2])

x = Concatenate()([x1, x2, x3, x4])
x = Dense(4 * hdims, kernel_regularizer="l2")(x)
x = Dropout(0.3)(x) # 模拟集成
x = gelu(x) # 有一点提升
outputs = Dense(num_classes, activation="softmax")(x)

model = Model([x1_input, x2_input], outputs)
model.summary()

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[0, 2, 6],
    values=[2*1e-3, 0.8*1e-3, 0.5*1e-3, 1e-4]
)
adam = tf.keras.optimizers.Adam(lr)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

model.fit([X1_train, X2_train], y_train, shuffle=True, batch_size=32, epochs=epochs, validation_split=0.1)

model_pooling_outputs = Model([x1_input, x2_input], [w1, w2])

id_to_classes = {j:i for i,j in classes.items()}
from textcolor import print_color_text
def visualization():
    for sample1, sample2, label in zip(X1_test, X2_test, y_test):
        sample1_len = len(sample1)
        sample2_len = len(sample2)

        x1 = np.array(tokenizer.transform([sample1]))
        x1 = sequence.pad_sequences(
            x1, 
            maxlen=maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )

        x2 = np.array(tokenizer.transform([sample2]))
        x2 = sequence.pad_sequences(
            x2, 
            maxlen=maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )

        y_pred = model.predict([x1, x2])[0]
        y_pred_id = np.argmax(y_pred)
        # 预测错误的样本跳过
        if y_pred_id != label:
            continue

        # 只看匹配的样本
        if y_pred_id != 1:
            continue

        # 预测权重
        w1, w2 = model_pooling_outputs.predict([x1, x2])
        w1 = w1[0]
        w2 = w2[0]
        w1 = w1.flatten()[:sample1_len]
        w2 = w2.flatten()[:sample2_len]
        print_color_text(sample1, w1)
        print()
        print_color_text(sample2, w2)
        print()

        input() # 按回车预测下一个样本

visualization()

