import numpy as np
import matplotlib.pyplot as plt
import functions as fn

def neural():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    print("X:", X.shape, X)
    print("W1:", W1.shape, W1)
    print("B1:", B1.shape, B1)

    A1 = np.dot(X, W1) + B1
    print("A1:", A1.shape, A1)
    Z1 = fn.sigmoid(A1)
    print("Z1:", Z1.shape, Z1)

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    print("W2:", W2.shape, W2)
    print("B2:", B2.shape, B2)

    A2 = np.dot(Z1, W2) + B2
    print("A2:", A2.shape, A2)
    Z2 = fn.sigmoid(A2)
    print("Z2:", Z2.shape, Z2)

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    print("W3:", W3.shape, W3)
    print("B3:", B3.shape, B3)

    A3 = np.dot(Z2, W3) + B3
    print("A3:", A3.shape, A3)
    Y = fn.identity_function(A3)
    print("Y:", Y.shape, Y)

if __name__ == "__main__":
    plt.title(__file__)
    plt.grid(True)

    neural()

    plt.show()
