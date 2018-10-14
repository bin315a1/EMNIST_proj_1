# EMNIST_proj_1
EMNIST reader in python with tensorflow
Modified mnist.py & dataset.py from tensorflow
Import the following 4 datasets from NIST, URL: https://www.nist.gov/itl/iad/image-group/emnist-dataset

From EMNIST-MNISTFORMAT, use 

emnist-byclass-train-images-idx3-ubyte.gz
emnist-byclass-train-labels-idx3-ubyte.gz
emnist-byclass-test-images-idx3-ubyte.gz
emnist-byclass-test-labels-idx3-ubyte.gz

Uses 62 classes for classification:
0~9: MNIST
10~35: Uppercase Alphabet
36~61: Lowercase Alphabet

dataset_gzipLoader used to load 4 gzip dataset files
EMNIST_main uses tensorflow & imported data to train&test
result data recorded on variationrecords.txt per run
  Variations in: learningrate, NeuralNetwork depth, NN width, etc
