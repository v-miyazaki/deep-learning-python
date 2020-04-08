import numpy as np
import time
from datetime import timedelta

def sum_1to1000():
    ret = 0
    for i in range(1, 1001):
        ret += i
    return ret

def sum_1to1000_nparray():
    a = np.ones(1000)
    b = np.arange(1,1001)
    return int(a.dot(b))
    pass

if __name__ == "__main__":
    start = time.time()
    print(sum_1to1000())
    elapsed = time.time() - start
    print ("sum_1to1000(): {} ".format(str(timedelta(seconds=elapsed))))

    start = time.time()
    print(sum_1to1000_nparray())
    elapsed = time.time() - start
    print ("sum_1to1000_nparray(): {} ".format(str(timedelta(seconds=elapsed))))
    pass
