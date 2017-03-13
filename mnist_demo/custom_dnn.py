import tensorflow  as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import util

tf.flags.DEFINE_string(
    "train_data_pattern", "/tmp/mnist/data/train.csv", "Input file path pattern used for training.")

tf.flags.DEFINE_string(
    "eval_data_pattern", "/tmp/mnist/data/test.csv", "Input file path pattern used for eval.")

tf.flags.DEFINE_string(
    "test_data_pattern", "/tmp/mnist/data/test_no_label.csv", "Input file path pattern used for testing.")

tf.flags.DEFINE_integer("batch_size", 100, "Batch size.")

tf.flags.DEFINE_integer("num_train_steps", 1000, "The number of steps to run training for.")

tf.flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")

FLAGS = tf.flags.FLAGS

NUM_FEATURE_COLUMNS = 784


def get_feature_columns():
    features = tf.contrib.layers.real_valued_column(
        'features', dimension=NUM_FEATURE_COLUMNS, default_value=0.0)
    return [features]


def make_input_fn(file_name, mode):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        num_columns = NUM_FEATURE_COLUMNS + 1
        num_epochs = None
        randomize_input = True
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        num_columns = NUM_FEATURE_COLUMNS + 1
        num_epochs = 1
        randomize_input = False
    else:
        num_columns = NUM_FEATURE_COLUMNS
        num_epochs = 1
        randomize_input = False

    def _input_fn():
        columns = util.read_tensors_from_csv(file_name, num_columns=num_columns, batch_size=FLAGS.batch_size,
                                             num_epochs=num_epochs, randomize_input=randomize_input)
        normalized_features = []
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            targets = tf.cast(columns[0], tf.int32)
            for i in range(1, NUM_FEATURE_COLUMNS + 1):
                normalized_features.append(columns[i] / 255.0)
            features = {'features': tf.stack(normalized_features, axis=1)}
        else:
            targets = None
            for i in range(0, NUM_FEATURE_COLUMNS):
                normalized_features.append(columns[i] / 255.0)
            features = {'features': tf.stack(normalized_features, axis=1)}
        return features, targets

    return _input_fn


def custom_model_fn(features, labels, mode, params):
    n_classes = params['n_classes']
    hidden_layer_units = params['hidden_layer_units']
    learning_rate = params['learning_rate']

    input_features = tf.contrib.layers.input_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=get_feature_columns())

    last_layer = input_features
    for layer_units in hidden_layer_units:
        last_layer = tf.contrib.layers.relu(last_layer, layer_units)

    logits = tf.contrib.layers.linear(last_layer, n_classes)

    predictions = tf.argmax(logits, 1)
    predictions_dict = {'labels': predictions}

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer="Adam")
        eval_metric_ops = None
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        train_op = None
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels, predictions)
        }
    elif mode == tf.contrib.learn.ModeKeys.INFER:
        loss = None
        train_op = None
        eval_metric_ops = None
    else:
        raise ValueError("Unexpected mode %d" % mode)

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def get_estimator(model_dir):
    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=1000)

    model_params = {'n_classes': 10, 'hidden_layer_units': [100, 100, 100], 'learning_rate': 0.001}

    tf.logging.info("Using Custom Estimator.")
    return tf.contrib.learn.Estimator(
        model_fn=custom_model_fn,
        params=model_params,
        model_dir=model_dir,
        config=config)


def train(model_dir):
    print('Training ...')

    estimator = get_estimator(model_dir)
    estimator.fit(input_fn=make_input_fn(FLAGS.train_data_pattern, tf.contrib.learn.ModeKeys.TRAIN),
                  steps=FLAGS.num_train_steps)

    print('Done.')


def eval(model_dir):
    print('Evaluating ...')

    estimator = get_estimator(model_dir)
    scores = estimator.evaluate(input_fn=make_input_fn(FLAGS.eval_data_pattern, tf.contrib.learn.ModeKeys.EVAL),
                                steps=None)
    print(scores)

    print('Done.')


def batch_predict(model_dir):
    print('Batch predicting ...')

    estimator = get_estimator(model_dir)
    labels = estimator.predict(input_fn=make_input_fn(FLAGS.test_data_pattern, tf.contrib.learn.ModeKeys.INFER))
    for score in labels:
        print(score)

    print('Done.')


def main():
    model_dir = '/tmp/mnist/model'

    train(model_dir)
    eval(model_dir)
    batch_predict(model_dir)


if __name__ == "__main__":
    main()
