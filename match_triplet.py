import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from attentionpooling import AttentionPooling1D
from dataset import SimpleTokenizer, find_best_maxlen
from dataset import load_lcqmc, convert_to_triplet

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

class MatchLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MatchLayer, self).__init__(**kwargs)
        self.o_dense = Dense(1)

    def call(self, inputs):
        x1, x2 = inputs
        # x*y
        x3 = Multiply()([x1, x2])
        # |x-y|
        x4 = Lambda(lambda x: tf.abs(x[0] - x[1]))([x1, x2])

        x = Concatenate()([x1, x2, x3, x4])
        x = self.o_dense(x)
        return x

class TripletLossLayer(tf.keras.layers.Layer):

    def __init__(self, similar, margin=10.0, **kwargs):
        super(TripletLossLayer, self).__init__(**kwargs)
        self.similar = similar # similar in a batch
        self.margin = tf.cast(margin, tf.float32)

    def call(self, inputs, mask=None):
        xa, xp, xn = inputs
        ploss = self.similar([xa, xp]) # 尽可能缩小
        nloss = self.similar([xa, xn]) # 尽可能增大
        loss = tf.math.maximum(ploss - nloss + self.margin, 0.0)
        self.add_loss(tf.reduce_mean(loss))
        return loss

Xa, Xp, Xn, classes = convert_to_triplet(load_lcqmc)
Xa_train = Xa[:-1000]
Xp_train = Xp[:-1000]
Xn_train = Xn[:-1000]

Xa_test = Xa[:-1000]
Xp_test = Xp[:-1000]
Xn_test = Xn[:-1000]

num_classes = len(classes)
tokenizer = SimpleTokenizer()
tokenizer.fit(Xa)

def pad(X, maxlen):
    X = sequence.pad_sequences(
        X, 
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0
    )
    return X

def create_dataset(Xa, Xp, Xn, maxlen):
    Xa = tokenizer.transform(Xa)
    Xp = tokenizer.transform(Xp)
    Xn = tokenizer.transform(Xn)

    Xa = pad(Xa, maxlen)
    Xp = pad(Xp, maxlen)
    Xn = pad(Xn, maxlen)
    return Xa, Xp, Xn

maxlen = find_best_maxlen(Xa)
maxlen = 48
hdims = 128
epochs = 1
num_words = len(tokenizer)
embedding_dims = 128

x1_input = Input(shape=(maxlen,))
x2_input = Input(shape=(maxlen,))
x3_input = Input(shape=(maxlen,))
# 计算全局mask
x1_mask = Lambda(lambda x: tf.not_equal(x, 0))(x1_input)
x2_mask = Lambda(lambda x: tf.not_equal(x, 0))(x2_input)
x3_mask = Lambda(lambda x: tf.not_equal(x, 0))(x3_input)

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

x3 = embedding(x3_input) + posembedding(x3_input)
x3 = layernom(x3)
x3 = Dropout(0.1)(x3)

conv1 = Conv1D(filters=hdims, kernel_size=2, padding="same", activation=gelu)
conv2 = Conv1D(filters=hdims, kernel_size=3, padding="same", activation=gelu)
conv3 = Conv1D(filters=hdims, kernel_size=4, padding="same", activation=gelu)

x1 = conv1(x1) + x1
x1 = conv2(x1) + x1
x1 = conv3(x1) + x1

x2 = conv1(x2) + x2
x2 = conv2(x2) + x2
x2 = conv3(x2) + x2

x3 = conv1(x3) + x3
x3 = conv2(x3) + x3
x3 = conv3(x3) + x3

pool = AttentionPooling1D(hdims)
x1, w1 = pool(x1, mask=x1_mask)
x2, w2 = pool(x2, mask=x2_mask)
x3, w3 = pool(x3, mask=x3_mask)

similar = MatchLayer()
outputs = TripletLossLayer(similar)([x1, x2, x3])

model = Model([x1_input, x2_input, x3_input], outputs)
model.summary()

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[0, 2, 6],
    values=[2*1e-3, 0.8*1e-3, 0.5*1e-3, 1e-4]
)
adam = tf.keras.optimizers.Adam(lr)
model.compile(optimizer="adam")
# model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

Xa_train, Xp_train, Xn_train = create_dataset(Xa_train, Xp_train, Xn_train, maxlen)

model.fit(
    [Xa_train, Xp_train, Xn_train],
    shuffle=True,
    batch_size=32,
    epochs=epochs,
    validation_split=0.1
)

model_pooling_outputs = Model([x1_input, x2_input, x3_input], [w1, w2, w3])

score = similar([x1, x2])
model_match_score_outputs = Model([x1_input, x2_input], score)

id_to_classes = {j:i for i,j in classes.items()}
from textcolor import print_color_text
def visualization():
    for xa, xp, xn in zip(Xa_test, Xp_test, Xn_test):
        xa_len = len(xa)
        xp_len = len(xp)
        xn_len = len(xn)

        x1 = np.array(tokenizer.transform([xa]))
        x1 = pad(x1, maxlen)

        x2 = np.array(tokenizer.transform([xp]))
        x2 = pad(x2, maxlen)

        x3 = np.array(tokenizer.transform([xn]))
        x3 = pad(x3, maxlen)

        y_pred = model.predict([x1, x2, x3])[0]

        score1 = model_match_score_outputs.predict([x1, x2])[0]
        score2 = model_match_score_outputs.predict([x1, x3])[0]

        # 预测权重
        w1, w2, w3 = model_pooling_outputs.predict([x1, x2, x3])
        w1 = w1[0]
        w2 = w2[0]
        w3 = w3[0]
        w1 = w1.flatten()[:xa_len]
        w2 = w2.flatten()[:xp_len]
        w3 = w3.flatten()[:xn_len]
        # print("triplet loss:", y_pred)
        print_color_text(xa, w1)
        print_color_text(xp, w2, withend=False)
        print("  match score:", score1)
        print_color_text(xn, w3, withend=False)
        print("  match score:", score2)
        print()

        input() # 按回车预测下一个样本

visualization()

