import sys, os
import numpy as np

# datasetディレクトリの参照を追加
dataset_dir = os.path.join(os.getcwd(), "git", "deep-learning-from-scratch")
# print(os.getcwd())
# print(dataset_dir)
sys.path.append(dataset_dir)
from dataset.mnist import load_mnist

def step_function(x):
    """ステップ関数

    :param a: a
    :type a: numpy array
    :return: y = int(x > 0)
    :rtype: numpy array
    """
    y = x > 0
    return y.astype(np.int)

def sigmoid(a):
    """Sigmoid function

    :param a: a
    :type a: numpy array
    :return: y = 1 / (1 + exp(-x))
    :rtype: numpy array
    """
    return 1 / (1 + np.exp(-a))

def identity_function(a):
    """恒等関数

    :param a: a
    :type a: numpy array
    :return: y = a
    :rtype: numpy array
    """
    return a

def softmax(a):
    """Softmax function

    :param a: a
    :type a: numpy array
    :return: y = [[exp(a(0)) / sum(exp(a))], ...]
    :rtype: numpy array
    """
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def mean_squared_error(y, t):
    """二乗和誤差

    :param y: ニューラルネットワークの出力
    :type y: numpy array
    :param t: 教師データ
    :type t: numpy array
    :return: 結果
    :rtype: numpy array
    """
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    """クロスエントロピー誤差

    :param y: ニューラルネットワークの出力
    :type y: numpy array
    :param t: 教師データ
    :type t: numpy array
    :return: 結果
    :rtype: numpy array
    """
    delta = 1e-7 # log(y)の計算結果が-infとならないようにする対応
    return -np.sum(t * np.log(y + delta))
