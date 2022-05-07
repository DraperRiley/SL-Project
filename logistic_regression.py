import math

import preprocess
import numpy as np

class LogisticRegression:

    def __init__(self, lr, image_width, image_height):
        self.lr = lr
        self.image_width = image_width
        self.image_height = image_height
        self.weights = np.random.rand(image_width*image_height + 1)
        self.cat_path = 'archive/training_set/training_set/cats/cat.{f}.jpg'
        self.dog_path = 'archive/training_set/training_set/dogs/dog.{f}.jpg'

    def predict(self, image_path):
        processor = preprocess.Preprocess(self.image_width, self.image_height, gray_scale=True)
        img = processor.process_image(image_path)
        img_array = np.array(img).flatten() / 255
        img_array = np.insert(img_array, 0, 1.0)
        dot_product = np.dot(self.weights, img_array)
        # print(dot_product)
        result = self.sigmoid(dot_product)
        return result, img_array

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.e**(-x))

    def train(self, epochs, cat_size, dog_size):

        while True:

            choose_bit = 0
            cat_index = 1
            dog_index = 1
            count = 0
            processor = preprocess.Preprocess(self.image_width, self.image_height)
            gradient_sum = np.zeros(self.image_width*self.image_height + 1)

            while count <= cat_size + dog_size:
                if choose_bit == 0:
                    path = self.cat_path.format(f=cat_index)
                    idx = cat_index
                    label = 0
                else:
                    path = self.dog_path.format(f=dog_index)
                    idx = dog_index
                    label = 1

                img = processor.process_image(path)
                img_array = np.array(img).flatten() / 255
                # img_array = np.array(img).flatten()
                img_array = np.insert(img_array, 0, 1)

                numerator = label * img_array
                label_times_weights = label * self.weights
                exponent = np.dot(label_times_weights.T, img_array)
                # print(exponent)
                denominator = 1 + math.e**exponent
                gradient_sum = gradient_sum + numerator / denominator
                #print(gradient_sum)

                if choose_bit == 0:
                    choose_bit = 1
                    cat_index += 1
                else:
                    choose_bit = 0
                    dog_index += 1

                count += 1

            gradient = -(1/(cat_size+dog_size)) * gradient_sum
            self.weights = self.weights - self.lr*gradient

            test_acc = self.test()
            print(test_acc)
            #print(self.weights)
            if test_acc >= 0.80:
                return

    def test(self):

        cat_test_path = 'archive/test_set/test_set/cats/cat.{f}.jpg'
        dog_test_path = 'archive/test_set/test_set/dogs/dog.{f}.jpg'

        dog_idx = 4001
        cat_idx = 4001

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        while dog_idx <= 5000:
            prediction, img_array = self.predict(dog_test_path.format(f=dog_idx))
            if prediction == 1:
                true_pos += 1
            else:
                false_neg += 1
            dog_idx += 1

        while cat_idx <= 5000:
            prediction, img_array = self.predict(cat_test_path.format(f=cat_idx))
            if prediction == 0:
                true_neg += 1
            else:
                false_pos += 1
            cat_idx += 1

        return (true_pos + true_neg) / 2000.0


def main():

    logistic_regression = LogisticRegression(0.5, 28, 28)
    #print(logistic_regression.weights)
    logistic_regression.train(10, 1000, 1000)
    #print(logistic_regression.weights)
    prediction, _ = logistic_regression.predict('archive/test_set/test_set/cats/cat.4002.jpg')
    print(prediction)

    return 0

if __name__ == '__main__':
    main()