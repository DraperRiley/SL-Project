import preprocess
import numpy as np
import matplotlib.pyplot as plt

# parameters
IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30
DOG_TRAIN_PATH = '../archive/training_set/training_set/dogs/dog.{f}.jpg'
CAT_TRAIN_PATH = '../archive/training_set/training_set/cats/cat.{f}.jpg'
DOG_TEST_PATH = '../archive/test_set/test_set/dogs/dog.{f}.jpg'
CAT_TEST_PATH = '../archive/test_set/test_set/cats/cat.{f}.jpg'
k_near = 9
dog = 1
cat = 0

'''
Data:
k = 3, acc = 0.5099009900990099
k = 5, acc = 0.5198019801980198
k = 7, acc = 0.5
k = 9, acc = 0.5396039603960396
'''

def main():

    # track predictions vs correct predictions
    predictions = 0
    correct_predictions = 0

    # test dog predictions
    test_index = 4001
    while test_index <= 4101:
        result = k_nn(DOG_TEST_PATH.format(f=test_index), k_near)
        test_index += 1
        predictions += 1
        if result == dog:
            correct_predictions += 1

    # test cat predictions
    test_index = 4001
    while test_index <= 4101:
        result = k_nn(CAT_TEST_PATH.format(f=test_index), k_near)
        test_index += 1
        predictions += 1
        if result == cat:
            correct_predictions += 1

    # print results
    print(correct_predictions / predictions)

    return 0


# main k-nearest neighbors function
def k_nn(img_path, k):

    # initialize preprocessor
    preprocessor = preprocess.Preprocess(IMAGE_WIDTH, IMAGE_HEIGHT, gray_scale=True)

    # lists for holding k-nearest neighbors and their labels
    k_min = []
    k_min_labels = []

    # initialize image array
    img = preprocessor.process_image(img_path)
    test_img_arr = np.array(img).flatten()

    # iterate through dog images, storing closest
    train_index = 1
    while train_index <= 500:

        img = preprocessor.process_image(DOG_TRAIN_PATH.format(f=train_index))
        train_img_arr = np.array(img).flatten()

        # get distance between images
        dist = get_distance(test_img_arr, train_img_arr)
        label = dog

        # increment
        train_index += 1

        # check if closer than current neighbors
        if len(k_min) < k:
            k_min.append(dist)
            k_min_labels.append(label)
        else:
            max_val = max(k_min)
            max_index = k_min.index(max_val)

            if dist < max_val:
                k_min[max_index] = dist
                k_min_labels[max_index] = label

    # reset index, iterate through cat images
    train_index = 1
    while train_index <= 500:

        img = preprocessor.process_image(CAT_TRAIN_PATH.format(f=train_index))
        train_img_arr = np.array(img).flatten()

        dist = get_distance(test_img_arr, train_img_arr)
        label = cat

        train_index += 1

        if len(k_min) < k:
            k_min.append(dist)
            k_min_labels.append(label)
        else:
            max_val = max(k_min)
            max_index = k_min.index(max_val)

            if dist < max_val:
                k_min[max_index] = dist
                k_min_labels[max_index] = label

    # check which label occurs more, return prediction
    k_val = k * 0.5
    if sum(k_min_labels) >= k_val:
        return 1
    else:
        return 0


# function to get distance between feature vectors
def get_distance(arr1, arr2):
    temp = arr1 - arr2
    temp = np.square(temp)
    result = np.sum(temp)
    return result

if __name__ == '__main__':
    x_val = ['k=3', 'k=5', 'k=7', 'k=9']
    y_val = [0.5099009900990099, 0.5198019801980198, 0.5, 0.5396039603960396]
    colors = ['green', 'blue', 'red', 'teal']
    plt.bar(x_val, y_val, color=colors)
    plt.title('Performance of K Near. Neigh.')
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('knn_performance.jpg')

    # main()
