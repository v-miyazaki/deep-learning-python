import os, sys
import pickle
import time
from datetime import timedelta

import numpy as np
from PIL import Image
from memory_profiler import profile

# datasetディレクトリの参照を追加
dataset_dir = os.path.join(os.getcwd(), "git", "deep-learning-from-scratch")
sys.path.append(dataset_dir)
from dataset.mnist import load_mnist

import functions as fn


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    path = os.path.join(os.getcwd(), "git", "deep-learning-from-scratch", "ch03", 'sample_weight.pkl')
    with open(path, 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = fn.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = fn.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = fn.softmax(a3)
    return y

@profile
def plain(x, t, network):
    start = time.time()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    elapsed = time.time() - start
    print ("time: {} ".format(str(timedelta(seconds=elapsed))))

@profile
def batch(x, t, network):
    start = time.time()
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1) # 最も確率の高い要素のインデックスを取得
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    elapsed = time.time() - start
    print ("time: {} ".format(str(timedelta(seconds=elapsed))))


x, t = get_data()
network = init_network()
plain(x, t, network)
batch(x, t, network)
