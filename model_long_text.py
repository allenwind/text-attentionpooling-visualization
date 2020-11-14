import random
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
from dataset import load_THUCNews_content_label

# 长文分类AttentionPooling可视化

gen, files, classes = load_THUCNews_content_label()
files = files[:]
num_classes = len(classes)
maxlen = 512

# cross validation
def split_index(size, scale=(0.8,0.1,0.1)):
    i = int(scale[0] * size)
    j = int((scale[0] + scale[1]) * size)
    return i, j

i, j = split_index(size=len(files))

files_train = files[:i]
files_val = files[i:j]
files_test = files[j:]

# train tokenizer
def Xiter(files):
    for content, label in gen(files):
        yield content

tokenizer = SimpleTokenizer()
tokenizer.fit(*[Xiter(files)])

class DataGenerator:

    def __init__(self, files, loop):
        self.files = files
        self.loop = loop

    def __call__(self):
        for _ in range(self.loop):
            random.shuffle(self.files)
            for content, label in gen(self.files):
                content = content[:maxlen]
                content = tokenizer.transform([content])[0]
                label = tf.keras.utils.to_categorical(label, num_classes)
                yield content, label

num_words = len(tokenizer)
embedding_dims = 128
batch_size = 32

def create_dataset(files, loop=int(1e12)):
    dataset = tf.data.Dataset.from_generator(
        generator=DataGenerator(files, loop),
        output_types=(tf.int32, tf.int32)
    )

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=([maxlen], [None]),
        drop_remainder=True
    )
    return dataset

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

model_weights_outputs = Model(inputs, w)

batch_size = 32
epochs = 2
callbacks = []

dl_train = create_dataset(files_train)
dl_val = create_dataset(files_val, loop=1)
model.fit(dl_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=dl_val,
          steps_per_epoch=len(files_train)//batch_size
)

id_to_classes = {j:i for i,j in classes.items()}
from color import print_color_string
def visualization():
    for sample, label in gen(files_test):
        sample_len = len(sample)
        if sample_len > maxlen:
            sample_len = maxlen

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
        weights = model_weights_outputs.predict(x)[0]
        # print(sample, "=>", id_to_classes[y_pred_id])
        # print(weights.flatten() * len(sample))

        weights = weights.flatten()[:sample_len]
        print_color_string(sample, weights)
        print(" =>", id_to_classes[y_pred_id])
        input() # 按回车预测下一个样本

visualization()
