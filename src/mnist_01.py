import os, sys
import numpy as np
from PIL import Image

# datasetディレクトリの参照を追加
dataset_dir = os.path.join(os.getcwd(), "git", "deep-learning-from-scratch")
sys.path.append(dataset_dir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False, one_hot_label=False)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(x_test.shape)


def printGraph(x):
    for i in range(28):
        str_line = ""
        for j in range(28):
            norm = int(x[i*28+j] / 256 * 10) # (0-255)->(0-9)に正規化
            str_line += str(norm)
        print(str_line)

# printGraph(x_train[0])

def img_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)
