import single_layer_perceptron
import preprocess
import numpy as np

def main():

    slp = single_layer_perceptron.SLP(0.2, 200, 200)
    slp.load_weights('slp_weights.npy')
    dog_idx = 4001
    cat_idx = 4001

    cat_path = 'archive/test_set/test_set/cats/cat.{f}.jpg'
    dog_path = 'archive/test_set/test_set/dogs/dog.{f}.jpg'

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    while dog_idx <= 5000:
        prediction, img_array = slp.predict(dog_path.format(f=dog_idx))
        if prediction == 1:
            true_pos += 1
        else:
            false_neg += 1
        dog_idx += 1

    while cat_idx <= 5000:
        prediction, img_array = slp.predict(cat_path.format(f=cat_idx))
        if prediction == -1:
            true_neg += 1
        else:
            false_pos += 1
        cat_idx += 1

    print(true_pos, true_neg, false_pos, false_neg)

    return 0

if __name__ == '__main__':
    main()