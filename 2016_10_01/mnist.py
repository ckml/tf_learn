import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

# Cross-entropy loss
def TrainKSVM(train_features, train_labels, valid_features, valid_labels,
               num_kfeatures, lamda, num_epochs, batch_size):
    num_instances = train_features.shape[0]
    num_classes = train_labels.shape[1]

    indices = np.arange(num_instances)
    np.random.shuffle(indices)
    indices = indices[0:num_kfeatures]

    kbase = train_features[indices, :]

    # RBF Kernel feature vectors for training
    ktrain_features = np.empty([num_instances, num_kfeatures])
    for i in range(num_instances):
        instance = train_features[i, :]
        instances = np.tile(instance, [num_kfeatures, 1])
        diff = instances - kbase
        diff = np.sum(diff * diff, axis=1)
        diff.shape = (1, num_kfeatures)
        ktrain_features[i, :] = diff

    print "Train instances converted."

    # RBF Kernel feature vectors for validation
    num_valid_instances = valid_features.shape[0]
    kvalid_features = np.empty([num_valid_instances, num_kfeatures])
    for i in range(num_valid_instances):
        instance = valid_features[i, :]
        instances = np.tile(instance, [num_kfeatures, 1])
        diff = instances - kbase
        diff = np.sum(diff * diff, axis=1)
        diff.shape = (1, num_kfeatures)
        kvalid_features[i, :] = diff

    print "Valid instances converted."

    kfeatures = tf.placeholder(tf.float32, shape=[None, num_kfeatures])
    label = tf.placeholder(tf.float32, shape=[None, num_classes])

    W = tf.Variable(tf.random_normal([num_kfeatures, num_classes]),
                    name="weights")
    b = tf.Variable(tf.random_normal([num_classes]), name="bias")
    lmda = tf.Variable(1.0, name="lambda")

    kfeatures2 = tf.exp(-lmda * kfeatures)
    net_output = tf.nn.softmax(tf.matmul(kfeatures2, W) + b) + 1e-10

    loss = -tf.reduce_sum(label * tf.log(net_output))
    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    prediction = tf.argmax(net_output, 1)
    is_correct = tf.equal(prediction, tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch_i in range(num_epochs):
        for batch_i in range(num_instances // batch_size):
            start_index = batch_i * batch_size
            stop_index = batch_i * batch_size + batch_size
            batch_features = ktrain_features[start_index: stop_index, :]
            batch_labels = train_labels[start_index: stop_index]
            sess.run(optimizer, feed_dict={
                kfeatures: batch_features,
                label: batch_labels
            })
        if epoch_i % 100 == 0:
            print('Epoch: ', epoch_i, sess.run(accuracy,
                                               feed_dict={
                                                   kfeatures: kvalid_features,
                                                   label: valid_labels
                                               }),
                  sess.run(accuracy,
                           feed_dict={
                               kfeatures: ktrain_features,
                               label: train_labels
                           }))
        else:
            print('Epoch: ', epoch_i, sess.run(accuracy,
                                               feed_dict={
                                                   kfeatures: kvalid_features,
                                                   label: valid_labels
                                               }))

data_folder = "/Users/thomasfu/data/mnist"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_features = mnist.train.images
train_labels = mnist.train.labels

valid_features = mnist.validation.images
valid_labels = mnist.validation.labels

test_features = mnist.test.images
test_labels = mnist.test.labels

TrainKSVM(train_features, train_labels, valid_features, valid_labels, 2000, 0.01, 10000, 100)