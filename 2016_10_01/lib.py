import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def zeroed_weight_variable(shape, var_name='', is_trainable=True):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name=var_name, trainable=is_trainable)


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


def CreateFM(num_features, num_classes, c=1.0, learning_rate=0.0001, cross_layers=0):
    tf.reset_default_graph()

    # type: (object, object, object, object) -> object
    features = tf.placeholder(tf.float32, shape=[None, num_features])
    labels = tf.placeholder(tf.float32, shape=[None, num_classes])

    input_feautres = features

    sum = 0
    for i in range(0, cross_layers):
        W_name = "W_layer_" + str(i)
        is_trainable = i == cross_layers - 1
        W = zeroed_weight_variable([num_features, 1], W_name, is_trainable)
        if is_trainable:
            sum = sum + tf.reduce_sum(tf.square(W))
        features2 = tf.matmul(features, W)
        features3 = features * features2 + features
        features = features3

    W2 = tf.Variable(tf.zeros([num_features, num_classes]),
                     name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")

    # Cross entropy loss
    net_output = tf.nn.softmax(tf.matmul(features, W2) + b)
    loss = -tf.reduce_sum(labels * tf.log(net_output + 1e-20)) + sum * c

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    prediction = tf.argmax(net_output, 1)
    is_correct = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

    vars = []
    for var in tf.all_variables():
        vars.append(var.op.name)

    model = {'features': input_feautres, 'labels': labels,
             'prediction': prediction,
             'accuracy': accuracy, 'optimizer': optimizer,
             'loss': loss, 'vars': vars}

    return model


def Train(model, train_data, valid_data, train_spec, max_accuracy):
    num_instances = train_data['features'].shape[0]

    num_epochs = train_spec['num_epochs']
    batch_size = train_spec['batch_size']
    stopping_threshold = train_spec['stopping_threshold']
    check_point_file = train_spec['checkpoint']

    # create a training session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # create a checkpoint saver
    if 'prev_vars' in train_spec:
        var_map = {}
        for var in tf.all_variables():
            if var.op.name in train_spec['prev_vars']:
                var_map[var.op.name] = var
        model_saver = tf.train.Saver(var_map)
        model_saver.restore(sess, train_spec['prev_checkpoint'])

    for var in tf.all_variables():
        is_trainable = var in tf.trainable_variables()
        if is_trainable:
            print var.op.name, 'trainable', var.eval(sess)
        else:
            print var.op.name, 'non-trainable', var.eval(sess)

    saver = tf.train.Saver()

    for epoch_i in range(num_epochs):
        indices = np.arange(num_instances)
        np.random.shuffle(indices)
        for batch_i in range(num_instances // batch_size):
            start_index = batch_i * batch_size
            stop_index = batch_i * batch_size + batch_size
            batch_indices = indices[start_index: stop_index]
            batch_features = train_data['features'][batch_indices, :]
            batch_labels = train_data['labels'][batch_indices, :]
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

        if (accuracy > max_accuracy):
            max_accuracy = accuracy
            # save checkpoint
            save_path = saver.save(sess, check_point_file)
            print 'Epoch: ', epoch_i, train_accuracy, accuracy, "Saved: ", save_path, max_accuracy
        else:
            print 'Epoch: ', epoch_i, train_accuracy, accuracy

        if accuracy < max_accuracy * stopping_threshold:
            break

    for var in tf.all_variables():
        print var.op.name, var.eval(sess)

    print 'Trained - ', max_accuracy, '\n'

    return max_accuracy