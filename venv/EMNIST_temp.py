# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import gzip
import numpy
tf.set_random_seed(777)  # for reproducibility

# The URLs where the MNIST data can be downloaded.
_TEST_DATA_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-test-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-test-labels-idx1-ubyte.gz'
_TRAIN_DATA_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-train-labels-idx1-ubyte.gz'

num_images = 731668

# def extract_images(filename, num_images):
#     """Extract the images into a numpy array.
#     Args:
#       filename: The path to an MNIST images file.
#       num_images: The number of images in the file.
#     Returns:
#       A numpy array of shape [number_of_images, height, width, channels].
#     """
#     print('Extracting images from: ', filename)
#     with gzip.open(filename) as bytestream:
#         bytestream.read(16)
#         buf = bytestream.read(
#             _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
#         data = np.frombuffer(buf, dtype=np.uint8)
#         data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
#     return data

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051: #2051
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    print(num_images)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(f, one_hot=False, num_classes=62):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    return labels

# def extract_labels(filename, num_labels):
#     """Extract the labels into a vector of int64 label IDs.
#     Args:
#       filename: The path to an MNIST labels file.
#       num_labels: The number of labels in the file.
#     Returns:
#       A numpy array of shape [number_of_labels]
#     """
#     print('Extracting labels from: ', filename)
#     with gzip.open(filename) as bytestream:
#         bytestream.read(8)
#         buf = bytestream.read(1 * num_labels)
#         labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#     return labels



with open(_TRAIN_DATA_FILENAME, 'rb') as f:
    train_images = extract_images(f)
with open(_TRAIN_LABELS_FILENAME, 'rb') as f:
    train_labels = extract_labels(f)
with open(_TEST_DATA_FILENAME, 'rb') as f:
    test_images = extract_images(f)
with open(_TEST_LABELS_FILENAME, 'rb') as f:
    test_labels = extract_labels(f)

nb_classes = 62

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
batch_size = 196

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int( 731668 / batch_size)    #total batch = 3733

        for i in range(total_batch):
            batch_xs, batch_ys = train_images, train_labels
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
