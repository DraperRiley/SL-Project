import numpy as np


def main():

    data = [1, 2, 3, 2.5]

    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]

    biases = [2, 3, 0.5]

    weights = np.array(weights)
    data = np.array(data)
    biases = np.array(biases)

    # These will be the same
    print(np.dot(weights, data) + biases)
    print(np.dot(data, weights.T) + biases)

    return 0


if __name__ == '__main__':
    main()
