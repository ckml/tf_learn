import tensorflow as tf
from utils import *
from lib import *
import numpy as np

def CreateModel():
    # %% Setup input to the network and true output label.  These are
    # simply placeholders which we'll fill in later.
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # %% Since x is currently [batch, height*width], we need to reshape to a
    # 4-D tensor to use it in a convolutional graph.  If one component of
    # `shape` is the special value -1, the size of that dimension is
    # computed so that the total size remains constant.  Since we haven't
    # defined the batch dimension's shape yet, we use -1 to denote this
    # dimension should not change size.
    x_tensor = tf.reshape(x, [-1, 28, 28, 1])

    # %% We'll setup the first convolutional layer
    # Weight matrix is [height x width x input_channels x output_channels]
    filter_size = 5
    n_filters_1 = 16
    W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

    # %% Bias is [output_channels]
    b_conv1 = bias_variable([n_filters_1])

    # %% Now we can build a graph which does the first layer of convolution:
    # we define our stride as batch x height x width x channels
    # instead of pooling, we use strides of 2 and more layers
    # with smaller filters.
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(input=x_tensor,
                     filter=W_conv1,
                     strides=[1, 2, 2, 1],
                     padding='SAME') +
        b_conv1)

    # %% And just like the first layer, add additional layers to create
    # a deep net
    n_filters_2 = 16
    W_conv2 = weight_variable(
        [filter_size, filter_size, n_filters_1, n_filters_2])
    b_conv2 = bias_variable([n_filters_2])
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(input=h_conv1,
                     filter=W_conv2,
                     strides=[1, 2, 2, 1],
                     padding='SAME') + b_conv2)

    # %% We'll now reshape so we can connect to a fully-connected layer:
    h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * n_filters_2])

    # %% Create a fully-connected layer:
    n_fc = 1024
    W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
    b_fc1 = bias_variable([n_fc])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    # %% We can add dropout for regularizing and to reduce overfitting like so:
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # %% And finally our softmax layer:
    W_fc2 = weight_variable([n_fc, 10])
    b_fc2 = bias_variable([10])
    y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # %% Define loss/eval/training functions
    cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    predicted = tf.argmax(y_pred, 1)

    # %% Monitor accuracy
    correct_prediction = tf.equal(predicted, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    model = {'x': x, 'y': y, 'y_pred': y_pred, 'accuracy': accuracy,
             'optimizer': optimizer, 'keep_prob': keep_prob}
    return model


def Train(model, data, batch_size, n_epochs, check_point_file):
    # create a checkpoint saver
    saver = tf.train.Saver()

    # %% We now create a new session to actually perform the initialization the
    # variables:
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # %% We'll train in minibatches and report accuracy:
    train_features = data['train_features']
    train_labels = data['train_labels']
    num_samples = train_labels.shape[0]

    valid_features = data['valid_features']
    valid_labels = data['valid_labels']

    optimizer = model['optimizer']

    max_accuracy = 0.0
    for epoch_i in range(n_epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for batch_i in range(num_samples // batch_size):
            start_index = batch_i * batch_size
            stop_index = batch_i * batch_size + batch_size
            batch_xs = train_features[indices[start_index: stop_index], :]
            batch_ys = train_labels[indices[start_index: stop_index], :]
            sess.run(optimizer, feed_dict={
                model['x']: batch_xs, model['y']: batch_ys,
                model['keep_prob']: 0.5})
        cur_accuracy = sess.run(model['accuracy'],
                                feed_dict={
                                    model['x']: valid_features,
                                    model['y']: valid_labels,
                                    model['keep_prob']: 1.0
                                })
        print('Epoch: ', epoch_i, cur_accuracy)
        if (cur_accuracy > max_accuracy):
            max_accuracy = cur_accuracy
            # save checkpoint
            saver.save(sess, check_point_file)
            print "Saved: ", max_accuracy

        if cur_accuracy < max_accuracy * 0.9:
            break


def BatchPredict(model, data, check_point_file):
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, check_point_file)

    batch_predicted = sess.run(model['y_pred'],
                               feed_dict={
                                   model['x']: data['features'],
                                   model['keep_prob']: 1.0
                               })
    return batch_predicted
