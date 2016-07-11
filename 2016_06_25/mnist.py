import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def TrainLL(train_features, train_labels, valid_features, valid_labels,
            num_epochs, batch_size, resume_training=False):
    num_instances = train_features.shape[0]
    num_features = train_features.shape[1]
    num_classes = train_labels.shape[1]

    features = tf.placeholder(tf.float32, shape=[None, num_features])
    label = tf.placeholder(tf.float32, shape=[None, num_classes])

    W = tf.Variable(tf.zeros([num_features, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")

    # Cross entropy loss
    net_output = tf.nn.softmax(tf.matmul(features, W) + b)
    loss = -tf.reduce_sum(label * tf.log(net_output))

    # Log loss
    # net_output = tf.matmul(features, W) + b
    # loss = tf.log(1.0 + tf.exp(-(label * 2 - 1) * net_output))

    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    prediction = tf.argmax(net_output, 1)
    is_correct = tf.equal(prediction, tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

    # register a variable to track.
    tf.scalar_summary("accuracy", accuracy)
    summary_op = tf.merge_all_summaries()

    # create a checkpoint saver
    saver = tf.train.Saver()

    # create a training session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # create a writer
    summary_writer = tf.train.SummaryWriter("/tmp/tf_trace3/", sess.graph)

    # resume from last checkpoint
    if resume_training:
        saver.restore(sess, "/tmp/model.ckpt")

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
        # save status
        if epoch_i % 10 == 0:
            summary_str = sess.run(summary_op, feed_dict={
                features: valid_features,
                label: valid_labels
            })
            summary_writer.add_summary(summary_str, epoch_i)
            summary_writer.flush()

            # save checkpoint
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)
    return sess, prediction, features

mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)

train_features = mnist.train.images
train_labels = mnist.train.labels

valid_features = mnist.validation.images
valid_labels = mnist.validation.labels

test_features = mnist.test.images
test_labels = mnist.test.labels

sess, prediction, features = TrainLL(train_features, train_labels,
                                     valid_features,
                                     valid_labels, 100, 100)

sess, prediction, features = TrainLL(train_features, train_labels,
                                     valid_features,
                                     valid_labels, 100, 100, True)
test_instance = test_features[50, :]
test_instance.shape = (1, 784)
predicted = sess.run(prediction, feed_dict={features: test_instance})
