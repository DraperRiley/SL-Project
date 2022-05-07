import math
import preprocess
import numpy as np

class SLP:

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
        if dot_product > 0:
            return 1, img_array
        else:
            return -1, img_array

    def train(self, epochs, cat_size, dog_size):

        for epoch in range(epochs):
            choose_bit = 0
            cat_index = 1
            dog_index = 1
            count = 0

            while count <= cat_size + dog_size:
                if choose_bit == 0:
                    path = self.cat_path.format(f=cat_index)
                    idx = cat_index
                    label = -1.0
                else:
                    path = self.dog_path.format(f=dog_index)
                    idx = dog_index
                    label = 1.0

                prediction, img_array = self.predict(path.format(f=idx))

                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * (label - prediction) * img_array[i]

                if choose_bit == 0:
                    choose_bit = 1
                    cat_index += 1
                else:
                    choose_bit = 0
                    dog_index += 1

                count += 1

        np.save('slp_weights.npy', self.weights)

    def load_weights(self, weight_path):
        self.weights = np.load(weight_path)


    def train_forever(self, cat_size, dog_size, threshold):

        epoch = 0

        while True:
            choose_bit = 0
            cat_index = 1
            dog_index = 1
            count = 0

            while count <= cat_size + dog_size:
                if choose_bit == 0:
                    path = self.cat_path.format(f=cat_index)
                    idx = cat_index
                    label = -1.0
                else:
                    path = self.dog_path.format(f=dog_index)
                    idx = dog_index
                    label = 1.0

                prediction, img_array = self.predict(path.format(f=idx))

                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * (label - prediction) * img_array[i]

                if choose_bit == 0:
                    choose_bit = 1
                    cat_index += 1
                else:
                    choose_bit = 0
                    dog_index += 1

                count += 1

            epoch += 1
            if epoch % 5 == 0:
                np.save('slp_weights_{f}_epochs.npy'.format(f=epoch), self.weights)
            acc = self.test()
            print(acc)
            if acc >= threshold:
                np.save('slp_weights_final.npy', self.weights)
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
            if prediction == -1:
                true_neg += 1
            else:
                false_pos += 1
            cat_idx += 1

        return (true_pos + true_neg) / 2000.0


def main():
    #slp = SLP(0.1, 200, 200)
    #print(slp.weights)
    #slp.train(2, 3999, 3999)
    #print(slp.weights)

    #slp = SLP(0.2, 200, 200)
    #slp.load_weights('slp_weights.npy')
    #print(slp.weights)

    slp_2 = SLP(0.3, 200, 200)
    slp_2.train_forever(3999, 3999, 0.85)

    #slp_2.train(5, 3999, 3999)
    #print(slp_2.weights)

    prediction, img_array = slp_2.predict('Cat03.jpg')
    print(prediction)

    return 0


if __name__ == '__main__':
    main()
