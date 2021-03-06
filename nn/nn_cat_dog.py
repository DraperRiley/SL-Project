import numpy as np
import nn
import preprocess
import matplotlib.pyplot as plt

# parameters
IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30
INPUTS = IMAGE_WIDTH*IMAGE_HEIGHT
layer1 = nn.Layer(INPUTS, 522)
layer2 = nn.Layer(522, 522)
output_layer = nn.Layer(522, 2)


'''
This is the main run of the ANN for image classification
'''

def main():

    # more parameters
    DOG_PATH = '../archive/training_set/training_set/dogs/dog.{f}.jpg'
    CAT_PATH = '../archive/training_set/training_set/cats/cat.{f}.jpg'
    lr = 0.0001
    iterations = 100
    dog = [[1, 0]]
    cat = [[0, 1]]
    target = None
    cat_size = 1000
    dog_size = 1000
    cat_index = 1
    dog_index = 1
    count = 0
    bit = 0
    prev_acc = 0

    # initialize helper classes
    preprocessor = preprocess.Preprocess(IMAGE_WIDTH, IMAGE_HEIGHT)
    loss_function = nn.Loss()
    backpropagation = nn.BackPropgation()

    # initialize lists for data collection
    loss = []
    x_val = []
    acc = []

    # for some number of iterations over entire image dataset
    for i in range(iterations):

        # initialize indices and loss
        cat_index = 1
        dog_index = 1
        count = 0
        loss_sum = 0

        # propagate images through network
        while count <= cat_size + dog_size:

            # switch between cat and dog images using bit variable
            if bit == 0:
                path = CAT_PATH.format(f=cat_index)
                cat_index += 1
                bit = 1
                target = cat
            else:
                path = DOG_PATH.format(f=dog_index)
                dog_index += 1
                bit = 0
                target = dog

            # flatten and normalize pixel data
            img = preprocessor.process_image(path)
            X = np.array(img).flatten() / 255
            X = np.array([X])

            # propagate vector through network
            layer1.forward(X)
            layer1_out = layer1.relu(layer1.output)

            layer2.forward(layer1_out)
            layer2_out = layer2.relu(layer2.output)

            output_layer.forward(layer2_out)
            pred = output_layer.sigmoid(output_layer.output)

            # sum loss
            loss_sum += loss_function.cumulative_loss(target, pred)

            # backpropagation to update weights
            delta_w, bias_update, delta_k = backpropagation.update_output(lr, output_layer, pred, target, layer2_out)
            output_layer.weights += delta_w
            output_layer.biases += bias_update

            delta_w, bias_update, delta_j = backpropagation.update_hidden(lr, layer2, output_layer, delta_k, layer1_out)
            layer2.weights += delta_w
            layer2.biases += bias_update

            delta_w, bias_update, delta_j = backpropagation.update_hidden(lr, layer1, layer2, delta_j, X)
            layer1.weights += delta_w
            layer1.biases += bias_update

            # iterate count
            count += 1

        # append loss and iteration
        loss.append(loss_sum)
        x_val.append(i)

        # test accuracy of network
        accuracy = test_acc()
        acc.append(accuracy)
        print('Iteration: {f}, Test acc: {g}'.format(f=i, g=accuracy))

    # calculate loss
    npy_loss = np.array(loss) / np.sum(loss)
    np_x_val = np.array(x_val)
    np_acc = np.array(acc)

    # plot data
    plt.plot(np_x_val, npy_loss, label='Loss')
    plt.plot(np_x_val, np_acc, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss/Accuracy')
    plt.title('NN Performance')
    plt.savefig('Loss_{f}_iter_{g}_lr.png'.format(f=iterations, g=lr))

    # save data for later use
    np.save('npy_loss_522_522_2.npy', npy_loss)
    np.save('npy_acc_522_522_2.npy', np_acc)
    np.save('np_x_val.npy', np_x_val)

    '''
    plt.plot(x_val, acc, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('NN Accuracy Over Test Data')
    plt.savefig('Acc_{f}_iter_{g}_lr.png'.format(f=iterations, g=lr))
    '''

    # np.save('layer1_weights.npy', layer1.weights)
    # np.save('layer1_biases.npy', layer1.biases)

    # np.save('layer2_weights.npy', layer2.weights)
    # np.save('layer2_biases.npy', layer2.biases)

    # np.save('output_layer_weights.npy', output_layer.weights)
    # np.save('output_layer_biases.npy', output_layer.biases)

    # make a prediction for testing purposes
    img = preprocessor.process_image('../archive/test_set/test_set/dogs/dog.4001.jpg')
    X = np.array(img).flatten() / 255

    layer1.forward(X)
    layer1_out = layer1.relu(layer1.output)

    layer2.forward(layer1_out)
    layer2_out = layer2.relu(layer2.output)

    output_layer.forward(layer2_out)
    pred = output_layer.sigmoid(output_layer.output)

    print(pred, 'Image was a dog')

    return 0


# a helper function for testing accuracy between iterations
def test_acc():

    # image paths for test data
    DOG_PATH = '../archive/test_set/test_set/dogs/dog.{f}.jpg'
    CAT_PATH = '../archive/test_set/test_set/cats/cat.{f}.jpg'

    # indexes
    cat_index = 4001
    dog_index = 4001

    # labels
    dog = 0
    cat = 1

    # predictions vs correct predictions
    predictions = 0
    correct_predictions = 0

    # initialize preprocessor
    preprocessor = preprocess.Preprocess(IMAGE_WIDTH, IMAGE_HEIGHT)

    # iterate through cat images
    while cat_index <= 5000:

        # get image
        img = preprocessor.process_image(CAT_PATH.format(f=cat_index))
        X = np.array(img).flatten() / 255
        X = np.array([X])

        # forward propagation
        layer1.forward(X)
        layer1_out = layer1.relu(layer1.output)

        layer2.forward(layer1_out)
        layer2_out = layer2.relu(layer2.output)

        output_layer.forward(layer2_out)
        pred = output_layer.sigmoid(output_layer.output)

        # get prediction label
        arg_pred = np.argmax(pred)

        # if correct, increment correct_predictions
        if arg_pred == cat:
            correct_predictions += 1

        # increment indices
        predictions += 1
        cat_index += 1

    # iterate similarly through dog test images
    while dog_index <= 5000:

        img = preprocessor.process_image(DOG_PATH.format(f=dog_index))
        X = np.array(img).flatten() / 255
        X = np.array([X])

        layer1.forward(X)
        layer1_out = layer1.relu(layer1.output)

        layer2.forward(layer1_out)
        layer2_out = layer2.relu(layer2.output)

        output_layer.forward(layer2_out)
        pred = output_layer.sigmoid(output_layer.output)

        arg_pred = np.argmax(pred)

        if arg_pred == dog:
            correct_predictions += 1

        predictions += 1
        dog_index += 1

    accuracy = correct_predictions / predictions
    # print(accuracy)

    return accuracy


if __name__ == '__main__':

    model1_acc = np.load('npy_acc_522_2.npy')
    model2_acc = np.load('npy_acc_522_522_2.npy')
    x_val = np.load('np_x_val.npy')

    plt.plot(x_val, model1_acc, label='Model 1')
    plt.plot(x_val, model2_acc, label='Model 2')
    plt.legend()
    plt.title('NN Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('NN_performance_acc_compare.jpg')

    model1_avg = np.mean(model1_acc)
    model2_avg = np.mean(model2_acc)

    print('Model 1 mean: {f}'.format(f=model1_avg))
    print('Model 2 mean: {f}'.format(f=model2_avg))

    # main()
