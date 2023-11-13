import numpy as np
import ipdb
import struct

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class SGD():

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, params, grad):
        for key in params:
            params[key] -= self.lr * grad[key]


def load_images(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(filename):
    with open(filename, 'rb') as file:
        magic, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
    return labels


def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return np.where(x>=0,1,0)


def softmax(x):
    # aviod overflow
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def dsoftmaxWithCrosEntropy(y, label):
    # a0 - y0
    return np.mean((y - label), axis=0, keepdims = True) 


def CrossEntropyLoss(y, l):

    b = y.shape[0]
    loss = 0.0
    idx = np.where(l == 1)[1]
    for i, num in enumerate(idx):
        q = y[i, num]
        # aviod numerical problem
        q = np.maximum(q, 1e-7)
        loss += -np.log(q)
    loss /= b
    return loss



def get_one_hot_label(y, num = 10):
    '''
    b: batch_size
    l: label 
    num: number of label
    '''
    b = y.shape[0]
    T = np.zeros((b, num))
    for i, row in enumerate(T):
        row[y[i]] = 1
    return T

def numerical_gradient(func, x):
    h = 1e-4  
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = func(x)  
        x[idx] = tmp_val - h
        fxh2 = func(x)  
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  
        it.iternext()

    return grad


if __name__ =="__main__":

    x = np.random.uniform(-1, 1, (3,3))
    y = np.array([  [1],
                    [1],
                    [2]])
    y = get_one_hot_label(y,num=3 )
    print(y)
    print(dsoftmaxWithCrosEntropy(x,y))