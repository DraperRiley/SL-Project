import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import preprocess

# Image height and width
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

def main():

    # This network uses the architecture described in the Tensorflow documentation:
    # https://www.tensorflow.org/tutorials/images/cnn

    # class_names = ['cat', 'dog']

    # Loading preprocessed images (grayscaled images)
    train_images = np.load('train_data_processed.npy')
    train_labels = np.load('train_label_processed.npy')
    test_images = np.load('test_data_processed.npy')
    test_labels = np.load('test_label_processed.npy')

    # Normalizing image data
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Creating convolution and pooling layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # model.summary()

    # Flatten into fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Keep track of models progress
    history = model.fit(train_images, train_labels, epochs=50,
                    validation_data=(test_images, test_labels))

    # Plot progress
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Performance')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('cnn_performance')

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(test_acc)

    return 0


def create_data():

    # Function for data preprocessing
    # NOTE: Images are not randomly sorted as data independence is assumed

    # Initialize preprocessor
    preprocessor = preprocess.Preprocess(IMAGE_WIDTH, IMAGE_HEIGHT, gray_scale=False)

    # Set indices for iterating through images
    cat_index = 1
    dog_index = 1

    # Image paths
    CAT_TRAIN_PATH = '../archive/training_set/training_set/cats/cat.{f}.jpg'
    DOG_TRAIN_PATH = '../archive/training_set/training_set/dogs/dog.{f}.jpg'
    CAT_TEST_PATH = '../archive/test_set/test_set/cats/cat.{f}.jpg'
    DOG_TEST_PATH = '../archive/test_set/test_set/dogs/dog.{f}.jpg'


    # initialize training data numpy arrays
    img = preprocessor.process_image(CAT_TRAIN_PATH.format(f=cat_index))
    train_data = np.array([img])
    cat_index += 1
    train_label = np.array([[0]])

    # append image and labels
    img = preprocessor.process_image(DOG_TRAIN_PATH.format(f=dog_index))
    train_data = np.append(train_data, [img], axis=0)
    dog_index += 1
    train_label = np.append(train_label, [[1]], axis=0)

    # iterate through cat images and append image data and labels to arrays
    while cat_index <= 1000:
        img = preprocessor.process_image(CAT_TRAIN_PATH.format(f=cat_index))
        train_data = np.append(train_data, [img], axis=0)
        train_label = np.append(train_label, [[0]], axis=0)

        img = preprocessor.process_image(DOG_TRAIN_PATH.format(f=cat_index))
        train_data = np.append(train_data, [img], axis=0)
        train_label = np.append(train_label, [[1]], axis=0)

        cat_index += 1

    # initialize arrays for validation data
    cat_index = 4001
    img = preprocessor.process_image(CAT_TEST_PATH.format(f=cat_index))
    test_data = np.array([img])
    test_label = np.array([[0]])

    # append image and label
    img = preprocessor.process_image(DOG_TEST_PATH.format(f=cat_index))
    test_data = np.append(test_data, [img], axis=0)
    test_label = np.append(test_label, [[1]], axis=0)
    cat_index += 1

    # iterate through validation images
    while cat_index <= 5000:
        img = preprocessor.process_image(CAT_TEST_PATH.format(f=cat_index))
        test_data = np.append(test_data, [img], axis=0)
        test_label = np.append(test_label, [[0]], axis=0)

        img = preprocessor.process_image(DOG_TEST_PATH.format(f=cat_index))
        test_data = np.append(test_data, [img], axis=0)
        test_label = np.append(test_label, [[1]], axis=0)

        cat_index += 1

    # Save data as .npy files for later use
    np.save('train_data_processed.npy', train_data)
    np.save('train_label_processed.npy', train_label)
    np.save('test_data_processed.npy', test_data)
    np.save('test_label_processed.npy', test_label)


if __name__ == '__main__':
    main()
    # create_data()
