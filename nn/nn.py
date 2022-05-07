import xmlrpc.client

import numpy as np
import preprocess
import matplotlib.pyplot as plt

np.random.seed(0)

class Layer:

    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def relu(self, x):
        return np.maximum(0, self.output)

    def soft_max(self, x):
        exp = np.exp(x)
        result = exp / np.sum(exp, axis=1, keepdims=True)
        return result

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_relu(self, x):
        return (x > 0).astype(int)

class Loss:
    def cross_entropy_loss(self, pred, y):
        clipped_values = np.clip(pred, 1e-7, 1-1e-7)
        log_values = clipped_values[range(len(pred)), y]
        return -np.log(log_values)

    def cumulative_loss(self, yhat, pred):
        target_minus_output = yhat - pred
        squared_losses = np.square(target_minus_output)
        squared_loss_sum = np.sum(squared_losses, axis=1, keepdims=True)
        square_loss_sum_final = np.sum(squared_loss_sum)
        return 0.5 * square_loss_sum_final

class BackPropgation:
    def backpropogation(self):
        pass

def main():

    DOG_PATH = '../archive/training_set/training_set/dogs/dog.1.jpg'
    CAT_PATH = '../archive/training_set/training_set/cats/cat.1.jpg'
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    INPUTS = IMAGE_WIDTH*IMAGE_HEIGHT
    lr = 0.1
    iterations = 5000

    X = np.array([[1,0,0,0,0,0,0,0],
         [0,1,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0],
         [0,0,0,1,0,0,0,0],
         [0,0,0,0,1,0,0,0],
         [0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,1]])

    X2 = np.array([[1,0,0,0,0,0,0,0]])

    layer1 = Layer(8, 3)
    layer2 = Layer(3, 8)

    loss_function = Loss()

    loss = []
    x_val = []

    for i in range(iterations):

        x_val.append(i)
        loss_sum = 0

        for j in range(len(X)):

            nn_input = np.array([X[j]])
            # print(nn_input)

            layer1.forward(nn_input)
            temp = layer1.relu(layer1.output)

            layer2.forward(temp)
            pred = layer2.sigmoid(layer2.output)

            # class_targets = [0,1,2,3,4,5,6,7] We will stick to one-hot encoding
            loss_sum += loss_function.cumulative_loss(nn_input, pred)

            # Backpropagation for the output layer
            delta_k = pred * (1-pred) * (nn_input - pred)
            delta_w = np.zeros(layer2.weights.shape)
            delta_w += delta_k
            delta_w = lr * delta_w * temp.T
            layer2.weights += delta_w
            layer2.biases += lr * delta_k

            # Backpropagation for the hidden layer NOTE: hidden layer uses ReLU

            delta_w = np.copy(layer1.weights)
            dk_wk = delta_k.T * delta_w
            k_sum = np.sum(dk_wk, axis=0)
            delta_j = layer1.d_relu(layer1.output)
            delta_j = delta_j * k_sum
            # use xij, if no previous layer, use input data
            delta_w = np.zeros(layer1.weights.shape)
            delta_w = delta_w + nn_input.T
            delta_w = lr * delta_j * delta_w
            layer1.weights += delta_w
            layer1.biases += lr * delta_j

        loss.append(loss_sum)


    plt.plot(x_val, loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('NN Performance')
    plt.show()

    layer1.forward(X)
    temp = layer1.relu(layer1.output)

    layer2.forward(temp)
    pred = layer2.sigmoid(layer2.output)
    print(pred, '\n')

    print(np.argmax(pred, axis=1))


    return 0


if __name__ == '__main__':
    main()
