import cv2
import numpy as np

class Preprocess:

    def __init__(self, image_width, image_height, gray_scale=True):
        self.image_width = image_width
        self.image_height = image_height
        self.gray_scale = gray_scale

    def process_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.gray_scale else cv2.IMREAD_COLOR)
        scaled_img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        return scaled_img

    def show_image(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)


def main():

    processor = Preprocess(200, 200, gray_scale=True)
    img = processor.process_image('archive/training_set/training_set/dogs/dog.1.jpg')
    #processor.show_image(img)
    array = np.array(img).flatten()
    print(array.shape)
    print(array/255)

    return 0

if __name__ == '__main__':
    main()