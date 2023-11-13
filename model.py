import numpy as np
from utils import *
import ipdb

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class MLP:
    def __init__(self,input_size, hidden_size, output_size=10):
        
        # row: input
        # column: output
        # b : output

        self.params = {}
        self.params['W1'] = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.params['b1'] = np.zeros((1,hidden_size))
        self.params['W2'] = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.params['b2'] = np.zeros((1,output_size))


    def forward(self, x):
        # forward-propagation

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        x = x @ W1 + b1
        x = relu(x)
        x = x @ W2 + b2
        # x = relu(x)

        y = softmax(x)

        return y
    
    def loss(self, x, l):

        y = self.forward(x)
        return CrossEntropyLoss(y, l)

    def gradient(self, x, label):
        # back-propagation

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # forward
        y0 = x @ W1 + b1
        y1 = relu(y0)
        y2 = y1 @ W2 + b2
        y = softmax(y2)

        # loss = CrossEntropyLoss(y, label)

        # backward
        grads = {}

        # dsce = dsoftmaxWithCrosEntropy(y, label)
        dsce =  (y-label) / x.shape[0]

        # TODO:check
        dr = drelu(y0)

        dW2 = y1.T @ dsce
        db2 = np.sum(dsce, axis=0, keepdims=True)

        dW1 = x.T @ (dsce @ W2.T * dr)
        db1 = np.sum((dsce @ W2.T * dr), axis=0, keepdims=True)

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return grads
    
    def numerical_gradient(self,x,l):

        loss = lambda _: self.loss(x, l)

        grads = {}

        grads['W1'] = numerical_gradient(loss, self.params['W1'])
        grads['b1'] = numerical_gradient(loss, self.params['b1'])
        grads['W2'] = numerical_gradient(loss, self.params['W2'])
        grads['b2'] = numerical_gradient(loss, self.params['b2'])

        return grads

    def accuracy(self, x, l):

        y = self.forward(x)
        y = np.argmax(y, axis=1)
        l = np.argmax(l, axis=1)

        accuracy = np.sum(y == l) / x.shape[0]
        return accuracy
    


if __name__ =="__main__":
    model = MLP(input_size=5, hidden_size=10, output_size=5)
    x = np.random.uniform(-1, 1, (5,5))
    y = np.array([  [0],
                    [1],
                    [4],
                    [2],
                    [3]])
    y = get_one_hot_label(y, num=5)
    print(y)

    for i in range(1000):
        grad = model.gradient(x, y)
        # grad = model.numerical_gradient(x, y)
        opt = SGD(learning_rate=0.01)
        opt.step(model.params, grad)

    p = model.forward(x)
    print(p)