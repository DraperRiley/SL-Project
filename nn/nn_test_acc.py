import numpy as np

import nn
import numpy
import matplotlib.pyplot as plt
import preprocess


def main():

    DOG_PATH = '../archive/training_set/training_set/dogs/dog.{f}.jpg'
    CAT_PATH = '../archive/training_set/training_set/cats/cat.{f}.jpg'

    cat_index = 1
    dog_index = 1
    IMAGE_WIDTH = 30
    IMAGE_HEIGHT = 30
    INPUTS = IMAGE_WIDTH*IMAGE_HEIGHT

    dog = 0
    cat = 1

    predictions = 0
    correct_predictions = 0

    preprocessor = preprocess.Preprocess(IMAGE_WIDTH, IMAGE_HEIGHT)

    layer1 = nn.Layer(INPUTS, 522)
    # layer2 = nn.Layer(10, 522)
    output_layer = nn.Layer(522, 2)

    layer1.weights = np.load('layer1_weights.npy')
    layer1.biases = np.load('layer1_biases.npy')

    output_layer.weights = np.load('output_layer_weights.npy')
    output_layer.biases = np.load('output_layer_biases.npy')

    while cat_index <= 1000:

        img = preprocessor.process_image(CAT_PATH.format(f=cat_index))
        X = np.array(img).flatten() / 255
        X = np.array([X])

        layer1.forward(X)
        layer1_out = layer1.relu(layer1.output)

        output_layer.forward(layer1_out)
        pred = output_layer.sigmoid(output_layer.output)

        arg_pred = np.argmax(pred)

        if arg_pred == cat:
            correct_predictions += 1

        predictions += 1
        cat_index += 1

    while dog_index <= 1000:

        img = preprocessor.process_image(DOG_PATH.format(f=dog_index))
        X = np.array(img).flatten() / 255
        X = np.array([X])

        layer1.forward(X)
        layer1_out = layer1.relu(layer1.output)

        output_layer.forward(layer1_out)
        pred = output_layer.sigmoid(output_layer.output)

        arg_pred = np.argmax(pred)

        if arg_pred == dog:
            correct_predictions += 1

        predictions += 1
        dog_index += 1

    print(correct_predictions / predictions)

    return 0


if __name__ == '__main__':
    main()
