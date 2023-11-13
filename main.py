import numpy as np
from utils import *
from model import MLP
import ipdb



batch_size = 512
epochs = 10000
learning_rate = 0.01
hidden_size= 50



def data_processing():

    x_train = './MNIST/raw/train-images-idx3-ubyte'
    y_train = './MNIST/raw/train-labels-idx1-ubyte'
    x_test = './MNIST/raw/t10k-images-idx3-ubyte'
    y_test = './MNIST/raw/t10k-labels-idx1-ubyte'

    x_train = load_images(x_train)
    y_train = load_labels(y_train)
    x_test = load_images(x_test)
    y_test = load_labels(y_test)

    x_train, x_test = x_train / 255.0, x_test / 255.0 

    x_train = x_train.reshape(-1,784)  
    x_test = x_test.reshape(-1,784)  

    y_train = get_one_hot_label(y_train) 
    y_test = get_one_hot_label(y_test) 

    return x_train, x_test, y_train, y_test

def main():

    x_train, x_test, y_train, y_test = data_processing()
    train_size = x_train.shape[0]

    model = MLP(input_size=784, hidden_size=hidden_size, output_size=10)
    opt = SGD(learning_rate=learning_rate)

    # training
    for epoch in range(epochs):
        # equial with "shuffle"

        indices = np.arange(train_size)
        np.random.shuffle(indices)
        
        # train in mini-batch
        for i in range(0, train_size, batch_size):

            batch_indice = indices[i:i + batch_size]
            x_batch = x_train[batch_indice]
            y_batch = y_train[batch_indice]

            grad = model.gradient(x_batch, y_batch)
            opt.step(model.params, grad)

        train_acc = model.accuracy(x_train, y_train)
        loss = model.loss(x_train, y_train)
        test_acc = model.accuracy(x_test, y_test)
        print(f"epoch:{epoch}, train_loss is {loss}; train_acc is :{train_acc}; test_acc is :{test_acc}")


if __name__ =="__main__":
    main()