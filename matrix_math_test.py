import numpy as np


def main():

    data = [[0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1]]

    data = np.array(data)

    filter = [[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]]

    filter = np.array(filter)


    print(np.dot(data, filter, ))

    return 0


if __name__ == '__main__':
    main()
