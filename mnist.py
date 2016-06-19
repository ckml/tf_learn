import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets, svm, linear_model, metrics

def Train(train_instances, train_labels,
          validation_instances, validation_labels, sess,
          n_epochs=10, batch_size = 100):
    n_input = train_instances.shape[1]
    n_output = train_labels.shape[1]
    net_input = tf.placeholder(tf.float32, [None, n_input])

    W = tf.Variable(tf.zeros([n_input, n_output]), name="weights")
    b = tf.Variable(tf.zeros([n_output]), name="bias")
    net_output = tf.nn.softmax(tf.matmul(net_input, W) + b)

    y_true = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_true * tf.log(net_output))

    prediction = tf.argmax(net_output, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_true, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    optimizer = tf.train.GradientDescentOptimizer(
        0.01).minimize(cross_entropy)

    sess.run(tf.initialize_all_variables())

    for epoch_i in range(n_epochs):
        for batch_i in range(train_instances.shape[0] // batch_size):
            start_index = batch_i * batch_size;
            stop_index = batch_i * batch_size + batch_size - 1;
            batch_xs = train_instances[start_index : stop_index, :]
            batch_ys = train_labels[start_index : stop_index, :]
            sess.run(optimizer, feed_dict={
                net_input: batch_xs,
                y_true: batch_ys
            })
        print('Epoch: ', epoch_i, sess.run(accuracy,
                                           feed_dict={
                                               net_input: validation_instances,
                                               y_true: validation_labels}))
    model = {"prediction": prediction, "net_input": net_input}
    return model

def Predict(test_instance, model, sess):
    label = (sess.run(model["prediction"],
                   feed_dict={model["net_input"]: test_instance}))[0]
    return label

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Train samples: ", mnist.train.num_examples,
      "Test samples: ", mnist.test.num_examples,
      "Validation samples: ", mnist.validation.num_examples)

train_instances = mnist.train.images
train_labels = mnist.train.labels
validation_instances = mnist.validation.images
validation_labels = mnist.validation.labels

sess = tf.Session()
model = Train(train_instances, train_labels,
              validation_instances, validation_labels, sess)

test_instance = mnist.test.images[0, :]
test_instance.shape=(1, 784)
test_label = np.argmax(mnist.test.labels[0])

predicted_label = Predict(test_instance, model, sess);

print("Predicted:", predicted_label, "Truth: ", test_label)


