import numpy as np
import tensorflow as tf


def CreateLM(num_features, num_classes, learning_rate=0.0001):
    features = tf.placeholder(tf.float32, shape=[None, num_features])
    labels = tf.placeholder(tf.float32, shape=[None, num_classes])

    W = tf.Variable(tf.zeros([num_features, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")

    # Cross entropy loss
    net_output = tf.nn.softmax(tf.matmul(features, W) + b)
    loss = -tf.reduce_sum(labels * tf.log(net_output))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    prediction = tf.argmax(net_output, 1)
    is_correct = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

    model = {'features': features, 'labels': labels, 'prediction': prediction,
             'accuracy': accuracy, 'optimizer': optimizer}

    return model


def Train(model, train_data, valid_data, num_epochs, batch_size,
          stopping_threshold=0.9, check_point_file='/tmp/model.ckpt',
          resume_training=False):
    num_instances = train_data['features'].shape[0]

    # create a checkpoint saver
    saver = tf.train.Saver()

    # create a training session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # resume from last checkpoint
    if resume_training:
        saver.restore(sess, check_point_file)

    max_accuracy = 0.0
    for epoch_i in range(num_epochs):
        indices = np.arange(num_instances)
        np.random.shuffle(indices)
        for batch_i in range(num_instances // batch_size):
            start_index = batch_i * batch_size
            stop_index = batch_i * batch_size + batch_size
            batch_indices = indices[start_index: stop_index]
            batch_features = train_data['features'][batch_indices, :]
            batch_labels = train_data['labels'][batch_indices]
            sess.run(model['optimizer'], feed_dict={
                model['features']: batch_features,
                model['labels']: batch_labels
            })
        train_accuracy = sess.run(model['accuracy'],
                                  feed_dict={
                                      model['features']: train_data[
                                          'features'],
                                      model['labels']: train_data[
                                          'labels']
                                  })
        accuracy = sess.run(model['accuracy'],
                            feed_dict={
                                model['features']: valid_data[
                                    'features'],
                                model['labels']: valid_data[
                                    'labels']
                            })
        print('Epoch: ', epoch_i, train_accuracy, accuracy)

        if (accuracy > max_accuracy):
            max_accuracy = accuracy
            # save checkpoint
            save_path = saver.save(sess, check_point_file)
            print "Saved: ", save_path, max_accuracy

        if accuracy < max_accuracy * stopping_threshold:
            break

    return model
