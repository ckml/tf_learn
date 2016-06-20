import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data

def TrainLL(train_features, train_labels, valid_features, valid_labels,
            num_epochs, batch_size):
    num_instances = train_features.shape[0]
    num_features = train_features.shape[1]
    num_classes = train_labels.shape[1]

    features = tf.placeholder(tf.float32, shape=[None, num_features])
    label = tf.placeholder(tf.float32, shape=[None, num_classes])

    W = tf.Variable(tf.zeros([num_features, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")
    net_output = tf.nn.softmax(tf.matmul(features, W) + b)

    loss = -tf.reduce_sum(label * tf.log(net_output))
    optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

    prediction = tf.argmax(net_output, 1)
    is_correct = tf.equal(prediction, tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch_i in range(num_epochs):
        indices = np.arange(num_instances)
        np.random.shuffle(indices)
        for batch_i in range(num_instances // batch_size):
            start_index = batch_i * batch_size
            stop_index = batch_i * batch_size + batch_size
            batch_indices = indices[start_index: stop_index]
            batch_features = train_features[batch_indices, :]
            batch_labels = train_labels[batch_indices]
            sess.run(optimizer, feed_dict={
                features: batch_features,
                label: batch_labels
            })
        print('Epoch: ', epoch_i, sess.run(accuracy,
                                           feed_dict={
                                               features: valid_features,
                                               label: valid_labels
                                           }))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_features = mnist.train.images
train_labels = mnist.train.labels

valid_features = mnist.validation.images
valid_labels = mnist.validation.labels

test_features = mnist.test.images
test_labels = mnist.test.labels

#TrainKSVM2(train_features, train_labels, valid_features, valid_labels, 500,
#           0.01, 10000, 200)
TrainLL(train_features, train_labels, valid_features, valid_labels, 100, 200)




