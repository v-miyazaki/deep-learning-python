<!-- <script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [["\\(","\\)"] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script> -->

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# 用語について

## numpy

Pythonの数値計算用ライブラリ。
行列演算(numpyではarrayという単位)を効率よく行うことが可能。

```bash
pip install numpy
```

```python
import numpy as np

x = np.array([1, 2, 3])
y = 2 * x
print(y) # [2, 4, 6]
```

## matplotlib

Pythonのグラフプロット用ライブラリ。
データの可視化に利用する。

```bash
pip install matplotlib
```

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-np.pi, np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

## パーセプトロン

パーセプトロンとは、複数の入力から単一の出力を行うモデル。
ニューラルネットワークの前駆となる考え方。

入力:$x_1$、$x_2$、出力:$y$ とした場合、以下の状態となるようなモデル。
$x_1$に対して任意の重み($w_1$、$w_2$)を付与したものが任意の値$\theta$を超えた場合、$y=1$、超えない場合、$y=0$。

$$
y = \begin{cases}
0 &\text{ } (w_1x_1 + w_2x_2 \leqq \theta) \\
1 &\text{ } (w_1x_1 + w_2x_2 > \theta) \\
\end{cases}
$$

これを発展して、$\theta=-b$とすることで右辺を$0$とする。$b$をバイアスと呼び、発火のしやすさを決定する。
$$
y = \begin{cases}
0 &\text{ } (b + w_1x_1 + w_2x_2 \leqq 0) \\
1 &\text{ } (b + w_1x_1 + w_2x_2 > 0) \\
\end{cases}
$$

0を境に値が急激に変動するため、このような関数を**ステップ関数**と呼ぶ。

このモデルの重みを適切に指定することで、AND回路やOR回路を構成することができる。

例: AND回路の場合

$
w_1 = 0.5, w_2 = 0.5, b = -0.7
$

とする。

$
y = \begin{cases}
0 &\text{ } (-0.7 + 0.5x_1 + 0.5x_2 \leqq 0) \\
1 &\text{ } (-0.7 + 0.5x_1 + 0.5x_2 > 0) \\
\end{cases}
$

となり、

$
y = \begin{cases}
0 &\text{ } (x_1=0, x_2=0) \\
0 &\text{ } (x_1=0, x_2=1) \\
0 &\text{ } (x_1=1, x_2=0) \\
1 &\text{ } (x_1=1, x_2=1) \\
\end{cases}
$

が導き出せる。
サンプルコードは以下のようになる。

```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron(x, w, b):
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def AND(x1, x2):
    return perceptron(x=np.array([x1, x2]), w=np.array([0.5, 0.5]), b=-0.7)

if __name__ == "__main__":
    print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))
    # 0 0 0 1
```

## 多層パーセプトロン

単純なパーセプトロンは1つの式で表現できる範囲までしか取り扱うことができない。
AND回路やOR回路、NAND回路は実装できるが、XOR回路の実装ができない。
(0 0 0 1 や 1 1 1 0 のような線形の変化は可能だが、0 1 1 0 のように非線形の変化に対応できない。)
このため、パーセプトロンの出力を多層化する(出力を次のパーセプトロンの入力とする)ことで解決する。

$x_1 \oplus x_2 = (x_1 | x_2) \cdot \overline{x_1 \cdot x_2}$ の論理回路を実現する。

サンプルコードは以下のようになる。

```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron(x, w, b):
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def AND(x1, x2):
    return perceptron(x=np.array([x1, x2]), w=np.array([0.5, 0.5]), b=-0.7)

def OR(x1, x2):
    return perceptron(x=np.array([x1, x2]), w=np.array([0.5, 0.5]), b=-0.2)

def NAND(x1, x2):
    return perceptron(x=np.array([x1, x2]), w=np.array([-0.5, -0.5]), b=0.7)

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == "__main__":
    print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))
    # 0 1 1 0
```

## ニューラルネットワーク

ニューラルネットワークは、パーセプトロンの解釈を拡大して任意の入力層、複数の中間層、出力層を利用できるモデルとなる。
また、パーセプトロンで使用していた0or1の出力に使用した関数は**活性化関数**と呼ばれる。
線形関数は多層化しても結果が変わらないため、活性化関数には非線形の関数が用いられる。

活性化関数として、以下のようなものが代表的。

* ステップ関数
   $y = \begin{cases} 1 &\text{ } (x>0) \\ 0 &\text{ } (x \leqq 0)\end{cases}$

* シグモイド関数
   $y = {1 \over 1+e^{-x}}$

* ReLU関数
   $y = \begin{cases} x &\text{ } (x>0) \\ 0 &\text{ } (x \leqq 0)\end{cases}$

## 行列の計算

numpyにより、行列の計算を高速に行うことができる。

```python
import numpy as np

# numpy配列の構造
A = np.array([1, 2, 3, 4])
print(A) # [1, 2, 3, 4]
print(np.ndim(A)) # 1 配列の次元数
print(A.shape) # (4,) 配列の構造

# ドット積
B = np.array([1, 2], [3, 4])
C = np.array([5, 6], [7, 8])
np.dot(B, C) # array([[19, 22], [43, 50]])
```

ニューラルネットワークの重み付き和は以下の式で表すことができる。
$$
A^{(n)} = XW^{(n)} + B^{(n)}
$$
nは何層目かを表す。

ここで、
$$
A^{(n)}=(a_1^{(n)} \quad a_2^{(n)} \quad a_3^{(n)}), \newline
X=(x_1 \quad x_2), \newline
B^{(n)}=(b_1^{(n)} \quad b_2^{(n)} \quad b_3^{(n)}), \newline
W^{(n)}=\begin{pmatrix} w_{11}^{(n)} & w_{21}^{(n)} & w_{31}^{(n)}\end{pmatrix}
$$
