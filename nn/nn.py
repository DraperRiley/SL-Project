import numpy as np
import preprocess
import matplotlib.pyplot as plt

np.random.seed(0)

# class which defines a layer in an artificial neural network
class Layer:

    # constructor
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        # self.biases = 0.10 * np.random.randn(1, num_neurons)

    # forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    # relu activation
    def relu(self, x):
        return np.maximum(0, self.output)

    # soft-max activation
    def soft_max(self, x):
        exp = np.exp(x)
        result = exp / np.sum(exp, axis=1, keepdims=True)
        return result

    # sigmoid activation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of relu function
    def d_relu(self, x):
        return (x > 0).astype(int)


# class which defines loss calculations
class Loss:

    # cross_entropy loss
    def cross_entropy_loss(self, pred, y):
        clipped_values = np.clip(pred, 1e-7, 1-1e-7)
        log_values = clipped_values[range(len(pred)), y]
        return -np.log(log_values)

    # cumulative loss
    def cumulative_loss(self, yhat, pred):
        target_minus_output = yhat - pred
        squared_losses = np.square(target_minus_output)
        squared_loss_sum = np.sum(squared_losses, axis=1, keepdims=True)
        square_loss_sum_final = np.sum(squared_loss_sum)
        return 0.5 * square_loss_sum_final


# class which defines backpropagation for both hidden and output layers
class BackPropgation:

    # update function for output layer
    def update_output(self, lr, layer, pred, target, inputs):

        delta_k = pred * (1-pred) * (target - pred)
        delta_w = np.zeros(layer.weights.shape)
        delta_w += delta_k
        delta_w = lr * delta_w * inputs.T
        bias_update = lr * delta_k

        return delta_w, bias_update, delta_k

    # update function for hidden layers
    # NOTE: non-generic, uses derivative relu function
    def update_hidden(self, lr, layer, forward_layer, delta_k, inputs):

        delta_w = np.copy(forward_layer.weights)
        dk_wk = delta_k * delta_w
        k_sum = np.sum(dk_wk, axis=1)
        delta_j = layer.d_relu(layer.output)
        delta_j = delta_j * k_sum
        # use xij, if no previous layer, use input data
        delta_w = np.zeros(layer.weights.shape)
        delta_w = delta_w + inputs.T
        delta_w = lr * delta_j * delta_w
        bias_update = lr * delta_j

        return delta_w, bias_update, delta_j


# main is used to test correctness of backpropagation
def main():

    # parameters and paths
    DOG_PATH = '../archive/training_set/training_set/dogs/dog.1.jpg'
    CAT_PATH = '../archive/training_set/training_set/cats/cat.1.jpg'
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    INPUTS = IMAGE_WIDTH*IMAGE_HEIGHT
    lr = 0.01  # learning rate
    iterations = 10000

    # initalize input for the auto-encoder problem as described in Machine Learning by Tom Mitchell
    X = np.array([[1,0,0,0,0,0,0,0],
         [0,1,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0],
         [0,0,0,1,0,0,0,0],
         [0,0,0,0,1,0,0,0],
         [0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,1]])

    # Unused one hot encoding
    # X2 = np.array([[1,0,0,0,0,0,0,0]])

    # instantiate layers
    layer1 = Layer(8, 10)
    layer2 = Layer(10, 7)
    layer3 = Layer(7, 10)
    output_layer = Layer(10, 8)

    # instantiate objects for loss and backpropagation
    loss_function = Loss()
    back_prop = BackPropgation()

    # initialize variables for holding data
    loss = []
    x_val = []

    # iterate
    for i in range(iterations):

        # append i to x_val list, set loss sum to 0
        x_val.append(i)
        loss_sum = 0

        # for each vector in X
        for j in range(len(X)):

            nn_input = np.array([X[j]])
            # print(nn_input)

            # forward propagation through layers
            layer1.forward(nn_input)
            layer1_out = layer1.relu(layer1.output)  # relu

            layer2.forward(layer1_out)  # using layer1 output
            layer2_out = layer2.relu(layer2.output)

            layer3.forward(layer2_out)
            layer3_out = layer3.relu(layer3.output)

            output_layer.forward(layer3_out)
            pred = output_layer.sigmoid(output_layer.output)  # get prediction from output layer using sigmoid

            # class_targets = [0,1,2,3,4,5,6,7] UNUSED: We will stick to one-hot encoding
            loss_sum += loss_function.cumulative_loss(nn_input, pred)  # get loss

            # Backpropagation for the output layer
            '''
            delta_k = pred * (1-pred) * (nn_input - pred)
            delta_w = np.zeros(layer2.weights.shape)
            delta_w += delta_k
            delta_w = lr * delta_w * layer1_out.T
            layer2.weights += delta_w
            layer2.biases += lr * delta_k
            '''

            # update weights for output layer
            delta_w, bias_update, delta_k = back_prop.update_output(lr, output_layer, pred, nn_input, layer3_out)
            output_layer.weights += delta_w
            output_layer.biases += bias_update

            # Backpropagation for the hidden layer NOTE: hidden layer uses ReLU
            '''
            delta_w = np.copy(layer2.weights)
            dk_wk = delta_k * delta_w
            k_sum = np.sum(dk_wk, axis=1)
            delta_j = layer1.d_relu(layer1.output)
            delta_j = delta_j * k_sum
            # use xij, if no previous layer, use input data
            delta_w = np.zeros(layer1.weights.shape)
            delta_w = delta_w + nn_input.T
            delta_w = lr * delta_j * delta_w
            layer1.weights += delta_w
            layer1.biases += lr * delta_j
            '''

            # update weights for each layer
            delta_w, bias_update, delta_j = back_prop.update_hidden(lr, layer3, output_layer, delta_k, layer2_out)
            layer3.weights += delta_w
            layer3.biases += bias_update

            delta_w, bias_update, delta_j = back_prop.update_hidden(lr, layer2, layer3, delta_j, layer1_out)
            layer2.weights += delta_w
            layer2.biases += bias_update

            delta_w, bias_update, delta_j = back_prop.update_hidden(lr, layer1, layer2, delta_j, nn_input)
            layer1.weights += delta_w
            layer1.biases += bias_update

        # append loss for graphing later
        loss.append(loss_sum)

    # plot data
    plt.plot(x_val, loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('NN Performance')
    plt.show()

    # test if network can make correct predictions
    layer1.forward(X)
    layer1_out = layer1.relu(layer1.output)

    layer2.forward(layer1_out)
    layer2_out = layer2.relu(layer2.output)

    layer3.forward(layer2_out)
    layer3_out = layer3.relu(layer3.output)

    output_layer.forward(layer3_out)
    pred = output_layer.sigmoid(output_layer.output)

    print(np.argmax(pred, axis=1))

    return 0


if __name__ == '__main__':
    main()
