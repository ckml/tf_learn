import time
import numpy as np
import tensorflow as tf

from inputs import *

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp/uci_census',
                    'Directory to write event logs and checkpoint.')
flags.DEFINE_string('train_data', '/Users/thomasfu/Downloads/adult.data',
                    'File pattern for training data.')
flags.DEFINE_integer('steps', 10000, 'Number of batches to train.')


def test():
    features, labels = inputs(FLAGS.train_data)

    # Create a supervisor.
    supervisor = tf.train.Supervisor(logdir=FLAGS.train_dir)

    step = 0
    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            step = step + 1
            print("step: ", step, sess.run([features, labels]))
            if step >= FLAGS.steps:
                supervisor.request_stop()

def train():
    features, labels = inputs(FLAGS.train_data)

    num_features = features.get_shape()[1].value
    num_classes = labels.get_shape()[1].value
    print("num_features: ", num_features)
    print("num_classes: ", num_classes)

    W = tf.Variable(tf.random_normal([num_features, num_classes]),
                    name="weights")
    b = tf.Variable(tf.random_normal([num_classes]), name="bias")

    net_output = tf.nn.softmax(tf.matmul(features, W) + b) + 1e-10

    loss = -tf.reduce_sum(labels * tf.log(net_output))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    prediction = tf.argmax(net_output, 1)
    is_correct = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

    # Create a supervisor.
    supervisor = tf.train.Supervisor(logdir=FLAGS.train_dir)

    step = 0
    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            step = step + 1
            result = sess.run([optimizer, accuracy])
            print("step: ", step, result[1])
            if step >= FLAGS.steps:
                supervisor.request_stop()


def main(_):
    train()


if __name__ == '__main__':
  tf.app.run()
