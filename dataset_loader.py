import gzip
import numpy
import random
import tensorflow as tf
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt

##gzip type dataset loader for EMNIST
##modified dataset.py & mnist.py from tensorflow library

class Dataset:

    def __init__(self,
               images_loc,
               labels_loc,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):

        with open(images_loc, 'rb') as f:
            images = self.extract_images(f)
        with open(labels_loc, 'rb') as f:
            labels = self.extract_labels(f,one_hot)

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
          raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)

        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                     images.shape[1] , images.shape[2])
            print(images.shape[0], images.shape[1], images.shape[2])

            images = numpy.transpose(images, (0,2,1))
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])

        if dtype == dtypes.float32:
          # Convert from [0, 255] -> [0.0, 1.0].
          images = images.astype(numpy.float32)
          images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples
        self._IMAGE_SIZE = 28
        self._fake_data = fake_data
        self._one_hot = one_hot

    def _read32(self, bytestream):
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_images(self, f):
        print('Extracting', f)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:  # 2051
                raise ValueError('Invalid magic number %d in image file: %s' %
                                 (magic, f))
            num_images = self._read32(bytestream)
            self._num_examples = num_images
            print(num_images)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = numpy.frombuffer(buf, dtype=numpy.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def dense_to_one_hot (self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = numpy.arange(num_labels) * num_classes
        labels_one_hot = numpy.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def extract_labels(self, f, one_hot=True, num_classes=62):
        print('Extracting', f)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in label file: %s' %
                                 (magic, f))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8)
            if one_hot:
              return self.dense_to_one_hot(labels, num_classes)
            return labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = numpy.arange(self._num_examples)
          numpy.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def test(self):
        r = random.randint(0, self._num_examples - 1)
        # na = (self._images[r:r + 1].reshape(28, 28))
        print(r)
        plt.imshow(
            (self._images[r:r + 1].reshape(28, 28)),
            cmap='Greys',
            interpolation='nearest')
        plt.show()
        print(self._labels[r:r+1])

def main():
    _TEST_DATA_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-test-images-idx3-ubyte.gz'
    _TEST_LABELS_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-test-labels-idx1-ubyte.gz'
    _TRAIN_DATA_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-train-images-idx3-ubyte.gz'
    _TRAIN_LABELS_FILENAME = 'D:\DataSet\EMNIST_MNISTFORMAT\gzip\emnist-byclass-train-labels-idx1-ubyte.gz'

    E = Dataset(_TRAIN_DATA_FILENAME, _TRAIN_LABELS_FILENAME,697932)
    batch_size = 19387
    batch_x, batch_y = E.next_batch(batch_size)
    E.test()

main()

# _NUM_CHANNELS = 1
#
# # The names of the classes.
# _CLASS_NAMES = [
#     'zero',
#     'one',
#     'two',
#     'three',
#     'four',
#     'five',
#     'size',
#     'seven',
#     'eight',
#     'nine',
#     'A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
#     'N', 'O', 'P', 'Q','R','S','T','U','V','W','X','Y','Z',
#     'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
#     'q','r','s','t','u','v','w','x','y','z'
# ]