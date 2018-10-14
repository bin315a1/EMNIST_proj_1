import tensorflow as tf
import random
import dataset_gzipLoader as D
import matplotlib.pyplot as plt
tf.set_random_seed(777)

##EMNIST dataset reader
##Dataset from https://www.nist.gov/itl/iad/image-group/emnist-dataset
##

class EMnist:

    sess = tf.InteractiveSession()

    def __init__(self, learningRate=0.35, batch_size=19387,
                 epoch_rep=10, NN_depth=4, NN_width=720,
                 dropOut_keeprate=0.7, tensorboard=False):

        _TEST_DATA_FILENAME = 'emnist-byclass-test-images-idx3-ubyte.gz'
        _TEST_LABELS_FILENAME = 'emnist-byclass-test-labels-idx1-ubyte.gz'
        _TRAIN_DATA_FILENAME = 'emnist-byclass-train-images-idx3-ubyte.gz'
        _TRAIN_LABELS_FILENAME = 'emnist-byclass-train-labels-idx1-ubyte.gz'
        self.num_train = 697932
        self.num_test =  116323

        self.data_train = D.Dataset(_TRAIN_DATA_FILENAME, _TRAIN_LABELS_FILENAME)
        self.data_test = D.Dataset( _TEST_DATA_FILENAME, _TEST_LABELS_FILENAME)

        self.learningRate = learningRate
        self.batch_size = batch_size
        self.epoch_rep = epoch_rep
        self.NN_depth = NN_depth
        self.NN_width = NN_width
        self.tensorboard = tensorboard
        self.dropOut_keeprate = dropOut_keeprate
        self.nb_classes = 62
        self._cost = 0

        self.batch_totAmt = 36

        self.WList = []
        self.bList = []
        self.layerList = []
        self.WList_hist = []
        self.bList_hist = []
        self.layerList_hist = []

    def train(self):
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.Y_ans = tf.placeholder(tf.float32, [None, self.nb_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        # With Tensorboard
        if (self.tensorboard):
            with tf.name_scope("layer1"):
                self.WList.append(tf.get_variable("W1", shape=[784, self.NN_width],
                                                  initializer=tf.contrib.layers.xavier_initializer()))
                self.bList.append(tf.Variable(tf.random_normal([self.NN_width])))
                self.layerList.append(tf.nn.relu(tf.matmul(self.X, self.WList[0]) + self.bList[0]))
                self.layerList[0] = tf.nn.dropout(self.layerList[0], keep_prob=self.keep_prob)

                self.WList_hist.append(tf.summary.histogram("weights1", self.WList[0]))
                self.bList_hist.append(tf.summary.histogram("bias1", self.bList[0]))
                self.layerList_hist.append(tf.summary.histogram("layer1", self.layerList[0]))

            for i in range(self.NN_depth):
                W_label = "W" + str(i + 2)
                b_label = "b" + str(i + 2)
                layer_label = "layer" + str(i + 2)

                if (i + 1 < self.NN_depth):
                    with tf.name_scope(layer_label):
                        self.WList.append(tf.get_variable(W_label, shape=[self.NN_width, self.NN_width],
                                                          initializer=tf.contrib.layers.xavier_initializer()))
                        self.bList.append(tf.Variable(tf.random_normal([self.NN_width])))
                        self.layerList.append(
                            tf.nn.relu(tf.matmul(self.layerList[i], self.WList[i + 1]) + self.bList[i + 1]))
                        self.layerList[i + 1] = tf.nn.dropout(self.layerList[i + 1], keep_prob=self.keep_prob)

                        self.WList_hist.append(tf.summary.histogram(W_label, self.WList[i + 1]))
                        self.bList_hist.append(tf.summary.histogram(b_label, self.bList[i + 1]))
                        self.layerList_hist.append(tf.summary.histogram(layer_label, self.layerList[i + 1]))
                else:
                    self.WList.append(tf.get_variable(W_label, shape=[self.NN_width, self.nb_classes],
                                                      initializer=tf.contrib.layers.xavier_initializer()))
                    self.bList.append(tf.Variable(tf.random_normal([self.nb_classes])))
                    self.Y_hypo = tf.matmul(self.layerList[-1], self.WList[-1]) + self.bList[-1]

                    self.WList_hist.append(tf.summary.histogram(W_label, self.WList[i + 1]))
                    self.bList_hist.append(tf.summary.histogram(b_label, self.bList[i + 1]))
                    hypo_hist = tf.summary.histogram("hypothesis", self.Y_hypo)

            with tf.name_scope("cost"):
                cost_crossEntropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_hypo, labels=self.Y_ans))
                # cost_crossEntropy = tf.reduce_mean(-tf.reduce_sum(self.Y_ans * tf.log(self.Y_hypo), axis=1))
                cost_summ = tf.summary.scalar("cost", cost_crossEntropy)

            with tf.name_scope("train"):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(
                    cost_crossEntropy)

            # global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
            # globalstep_num = tf.train.global_step(self.sess, global_step_tensor)

            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(".\mnist_log_1")
            writer.add_graph(self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

            for epoch in range(self.epoch_rep):
                print("epoch #", epoch)
                for iteration in range(self.batch_totAmt):
                    batch_Xs, batch_Ys = self.data_train.next_batch(self.batch_size)  # where dataset feeds batches

                    summary, _ = self.sess.run([merged_summary, optimizer],
                                               feed_dict={self.X: batch_Xs, self.Y_ans: batch_Ys,
                                                          self.keep_prob: self.dropOut_keeprate})
                    writer.add_summary(summary)
        # without tensorboard
        else:  # without tensorboard
            self.WList.append(
                tf.get_variable("W1", shape=[784, self.NN_width], initializer=tf.contrib.layers.xavier_initializer()))
            self.bList.append(tf.Variable(tf.random_normal([self.NN_width])))
            self.layerList.append(tf.nn.relu(tf.matmul(self.X, self.WList[0]) + self.bList[0]))
            self.layerList[0] = tf.nn.dropout(self.layerList[0], keep_prob=self.keep_prob)

            for i in range(self.NN_depth):
                W_label = "W" + str(i + 2)
                b_label = "b" + str(i + 2)
                layer_label = "layer" + str(i + 2)

                if (i + 1 < self.NN_depth):
                    self.WList.append(tf.get_variable(W_label, shape=[self.NN_width, self.NN_width],
                                                      initializer=tf.contrib.layers.xavier_initializer()))
                    self.bList.append(tf.Variable(tf.random_normal([self.NN_width])))
                    self.layerList.append(
                        tf.nn.relu(tf.matmul(self.layerList[i], self.WList[i + 1]) + self.bList[i + 1]))
                    self.layerList[i + 1] = tf.nn.dropout(self.layerList[i + 1], keep_prob=self.keep_prob)

                else:
                    self.WList.append(tf.get_variable(W_label, shape=[self.NN_width, self.nb_classes],
                                                      initializer=tf.contrib.layers.xavier_initializer()))
                    self.bList.append(tf.Variable(tf.random_normal([self.nb_classes])))
                    self.Y_hypo = tf.matmul(self.layerList[-1], self.WList[-1]) + self.bList[-1]

            cost_crossEntropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_hypo, labels=self.Y_ans))
            # cost_crossEntropy = tf.reduce_mean(-tf.reduce_sum(self.Y_ans * tf.log(self.Y_hypo), axis=1))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(cost_crossEntropy)

            self.sess.run(tf.global_variables_initializer())

            for epoch in range(self.epoch_rep):
                avg_cost = 0
                for iteration in range(self.batch_totAmt):
                    batch_Xs, batch_Ys = self.data_train.next_batch(self.batch_size)
                    c, _ = self.sess.run([cost_crossEntropy, optimizer],
                                         feed_dict={self.X: batch_Xs, self.Y_ans: batch_Ys,
                                                    self.keep_prob: self.dropOut_keeprate})  # where x&y are feeded
                    avg_cost += c / self.batch_totAmt
                    self._cost = avg_cost
                print("epoch #", epoch, "cost =", '{:.9f}'.format(avg_cost))

    def result_summ(self):
        correctID = tf.equal(tf.argmax(self.Y_hypo, 1), tf.argmax(self.Y_ans,
                                                                  1))  # argmax: returns the index of the biggest number across the axis
        accuracy_run = tf.reduce_mean(tf.cast(correctID, tf.float32))
        self.accuracy = self.sess.run(accuracy_run,
                                      feed_dict={self.X: self.data_test._images, self.Y_ans: self.data_test._labels,
                                                 self.keep_prob: 1})
        print('_' * 80)
        print("Summary:\nLearning Rate:", self.learningRate, "\nBatch Size:", self.batch_size, "\nEpoch Repetition:",
              self.epoch_rep)
        print("Layer Count:", self.NN_depth + 1, "\nNN Width: ", self.NN_width, "\nDropout Keep Rate:",
              self.dropOut_keeprate)
        print("\nAccuracy:", format(self.accuracy, '.2%'))
        print('_' * 80)

        file = open("VariationRecords.txt", "a")
        file.write("\nTest 1:\n")
        file.write("LR: " + str(self.learningRate) + "\n")
        file.write("BS: " + str(self.batch_size) + '\n')
        file.write("ER: " + str(self.epoch_rep) + '\n')
        file.write("LC: " + str(self.NN_depth + 1) + '\n')
        file.write("NW: " + str(self.NN_width) + '\n')
        file.write("DR: " + str(self.dropOut_keeprate) + '\n')
        file.write("Accuracy: " + str(format(self.accuracy, '.2%')) + '\n')
        file.write("Cost: " + str(self._cost) + '\n')
        file.close()

    def test_run(self):
        r = random.randint(0, self.num_test - 1)
        print("Label for random selected sample : ", self.sess.run(tf.argmax(self.data_test._labels[r:r + 1], 1)))
        print("Machine prediction : ", self.sess.run(
            tf.argmax(self.Y_hypo, 1), feed_dict={self.X: self.data_test._images[r:r + 1], self.keep_prob: 1}))


        plt.imshow(
            self.data_test._images[r:r + 1].reshape(28, 28),
            cmap='Greys',
            interpolation='nearest')
        plt.show()


def main():
    sampleA = EMnist()
    sampleA.train()
    sampleA.test_run()
    sampleA.test_run()
    sampleA.result_summ()

main()
