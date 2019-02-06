# Lab 7 Learning rate and Evaluation
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import Dataset_main
tf.set_random_seed(777)  # for reproducibility

from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

_TEST_DATA_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-balanced-test-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-balanced-test-labels-idx1-ubyte.gz'
_TRAIN_DATA_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-balanced-train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-balanced-train-labels-idx1-ubyte.gz'

with open('my/directory/train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)
with open('my/directory/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = extract_images(f)

with open('my/directory/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = extract_labels(f)
with open('my/directory/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = extract_labels(f)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()