# Image classification using SL algorithms

This project aims to compare various supervised learning models when applied to 
image classification. Our [dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) consists of images of cats and dogs and is available of Kaggle.
Models used include K-Nearest Neighbors, Artificial Neural Networks, and Convolutional Neural Networks.

## KNN

K-Nearest Neighbors is a non-learning based algorithms which aims to predict a label for a given data vector using its distance between its K nearest neighbors.
Various values of K were used such as 3, 5, 7, and 9 and their accuracies compared, the results are as follows:

![KNN](/knn/knn_performance.jpg)

It was hypothesized that K=5 would provide the highest accuracy over the test data, with diminishing returns for higher values of K,
however, the highest value of K=9 provides the highest accuracy.

## ANN

Artificial Neural Networks are complex function approximators which can potentially solve both classification and regression problems.
They make use of activation functions which allow the approximation of non-linear functions. Two models were trained with varying architectures.
Model 1 one used an architecture of 900x522x2, whereas Model 2 used an architecture of 900x522x522x2. Both models use a learning rate of 0.0001, ReLU activation for the hidden layers, and Sigmoid activation on the output layer.
The results are described below:

![ANN](/nn/NN_performance_acc_compare.jpg)

It was hypothesized that Model 1 would have a higher mean accuracy than Model 2.
However, Model 1 achieved a mean accuracy of 56.4995%, and Model 2 an accuracy of 57.218%.

## CNN

Using Tensorflow, a Convolutional Neural Network was trained. CNNs succeed in image classification
due to their ability to maintain the context of the image, whereas ANNs used flattened pixel data.
The architecture used for the CNN is described [here](https://www.tensorflow.org/tutorials/images/cnn).
The results for the CNN are described below:

![CNN](/cnn/cnn_performance.png)

We can see that the CNN achieves much higher accuracy over the test_data than both KNN and ANNs. As CNNs are data hungry models, with more images
an even higher accuracy can surely be achieved.

## Conclusion

Due the high non-linearity of images, it was expected that both KNN and ANNs would
severely underperform comparedn to CNNs. In the future I would like to see how other more
complex models such as Support Vector Machines are able to handle this problem.