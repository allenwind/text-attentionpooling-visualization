import re
import random
import itertools
import collections
import pandas as pd
import numpy as np

_THUCNews = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"
def load_THUCNews_title_label(file=_THUCNews):
    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    titles = []
    labels = []
    for line in lines:
        title, label = line.split("\t")
        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label:i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

_w100k = "/home/zhiwen/workspace/dataset/classification/weibo_senti_100k/weibo_senti_100k.csv"
def load_weibo_senti_100k(file=_w100k, noe=True):
    df = pd.read_csv(file)
    X = df.review.to_list()
    y = df.label.to_list()
    # 去 emoji 表情
    if noe:
        X = [re.sub("\[.+?\]", lambda x:"", s) for s in X]
    categoricals = {"负面":0, "正面":1}
    return X, y, categoricals

_MOODS = "/home/zhiwen/workspace/dataset/classification/simplifyweibo_4_moods.csv"
def load_simplifyweibo_4_moods(file=_MOODS):
    df = pd.read_csv(file)
    X = df.review.to_list()
    y = df.label.to_list()
    categoricals = {"喜悦":0, "愤怒":1, "厌恶":2, "低落":3}
    return X, y, categoricals

_LCQMC = "/home/zhiwen/workspace/dataset/LCQMC/totals.txt"
def load_lcqmc(file=_LCQMC):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        x1, x2, label = line.strip().split("\t")
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

class SimpleTokenizer:
    """字转ID
    """

    def __init__(self, min_freq=16):
        self.char2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.min_freq = min_freq

    def fit(self, X):
        # 建立词ID映射表
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1
        # 过滤低频词
        chars = {i:j for i, j in chars.items() if j >= self.min_freq}
        # 0:MASK
        # 1:UNK
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNKNOW))
            ids.append(s)
        return ids

    def fit_transform(self, X):
        self.fit(X)
        ids = self.transform(X)
        return ids

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

    @property
    def vocab(self):
        return self.char2id

def find_best_maxlen(X, mode="max"):
    # 获取适合的截断长度
    ls = [len(sample) for sample in X]
    if mode == "mode":
        maxlen = np.argmax(np.bincount(ls))
    if mode == "mean":
        maxlen = np.mean(ls)
    if mode == "median":
        maxlen = np.median(ls)
    if mode == "max":
        maxlen = np.max(ls)
    return int(maxlen)
