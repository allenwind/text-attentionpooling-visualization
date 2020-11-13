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

from dataset import load_THUCNews_title_label
from dataset import load_weibo_senti_100k
from dataset import load_simplifyweibo_4_moods

X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=7384672)

num_classes = len(classes)
tokenizer = SimpleTokenizer()
tokenizer.fit(X_train)
X_train = tokenizer.transform(X_train)

# maxlen = 48
maxlen = find_best_maxlen(X_train)

X_train = sequence.pad_sequences(
    X_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0
)
y_train = tf.keras.utils.to_categorical(y_train)

num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(num_words, embedding_dims,
    embeddings_initializer="glorot_normal",
    input_length=maxlen)(inputs)
x = Dropout(0.2)(x)
x = Conv1D(filters=128,
           kernel_size=3,
           padding="same",
           activation="relu",
           strides=1)(x)
x, w = AttentionPooling1D(hdims=128)(x, mask=mask)
x = Dense(128)(x)
x = Dropout(0.2)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()

model_w_outputs = Model(inputs, w)

batch_size = 32
epochs = 10
callbacks = []
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_split=0.2)

id_to_classes = {j:i for i,j in classes.items()}
from color import print_color_string
def visualization():
    for sample, label in zip(X_test, y_test):
        sample_len = len(sample)
        x = np.array(tokenizer.transform([sample]))
        x = sequence.pad_sequences(
            x, 
            maxlen=maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )

        y_pred = model.predict(x)[0]
        y_pred_id = np.argmax(y_pred)
        # 预测错误的样本跳过
        if y_pred_id != label:
            continue
            
        # 预测权重
        weights = model_w_outputs.predict(x)[0]
        # print(sample, "=>", id_to_classes[y_pred_id])
        # print(weights.flatten() * len(sample))

        weights = weights.flatten()[:sample_len]
        print_color_string(sample, weights)
        print(" =>", id_to_classes[y_pred_id])
        input() # 按回车预测下一个样本

visualization()

