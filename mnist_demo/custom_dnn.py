import functools
import tensorflow  as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.training import input as input_lib
import util

tf.flags.DEFINE_string(
    "train_data_pattern", "/Users/thomasfu/data/mnist/train.csv", "Input file path pattern used for training.")

tf.flags.DEFINE_string(
    "eval_data_pattern", "/Users/thomasfu/data/mnist/test.csv", "Input file path pattern used for eval.")

tf.flags.DEFINE_string(
    "test_data_pattern", "/Users/thomasfu/data/mnist/test_no_label.csv", "Input file path pattern used for testing.")

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
                normalized_features.append(columns[i])
            features = {'features': tf.stack(normalized_features, axis=1)}
        else:
            targets = None
            for i in range(0, NUM_FEATURE_COLUMNS):
                normalized_features.append(columns[i])
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
    scores = tf.nn.softmax(logits)

    predictions = tf.argmax(logits, 1)
    predictions_dict = {'labels': predictions, 'scores': scores}

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

def feature_engineering_fn(features, labels):
    for feature in features:
        features[feature] /= 255.0
    return features, labels

def get_estimator(model_dir):
    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=1000)

    model_params = {'n_classes': 10, 'hidden_layer_units': [100, 100, 100], 'learning_rate': 0.001}

    tf.logging.info("Using Custom Estimator.")
    return tf.contrib.learn.Estimator(
        model_fn=custom_model_fn,
        params=model_params,
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


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

def make_online_prediction_input_fn(instances):
    def _input_fn(instances):
        features = {
            'features':
                input_lib.limit_epochs(tf.constant(instances), num_epochs=1)
        }
        return features, None
    return functools.partial(_input_fn, instances=instances)


def online_predict(model_dir):
    print('Online predicting ...')

    estimator = get_estimator(model_dir)

    instances = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240,
         253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253,
         253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253,
         187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 159, 253, 159, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 238, 252, 252, 252, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 54, 227, 253, 252, 239, 233, 252, 57, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 60,
         224, 252, 253, 252, 202, 84, 252, 253, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 252, 252,
         252, 253, 252, 252, 96, 189, 253, 167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 238, 253, 253, 190,
         114, 253, 228, 47, 79, 255, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 238, 252, 252, 179, 12, 75,
         121, 21, 0, 0, 253, 243, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 165, 253, 233, 208, 84, 0, 0, 0, 0, 0,
         0, 253, 252, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 178, 252, 240, 71, 19, 28, 0, 0, 0, 0, 0, 0, 253, 252,
         195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 252, 252, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 252, 195, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 253, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 253, 196, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 76, 246, 252, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 252, 148, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85,
         252, 230, 25, 0, 0, 0, 0, 0, 0, 0, 0, 7, 135, 253, 186, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 223, 0,
         0, 0, 0, 0, 0, 0, 0, 7, 131, 252, 225, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 145, 0, 0, 0, 0, 0, 0,
         0, 48, 165, 252, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 253, 225, 0, 0, 0, 0, 0, 0, 114, 238, 253,
         162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 249, 146, 48, 29, 85, 178, 225, 253, 223, 167, 56,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 252, 252, 229, 215, 252, 252, 252, 196, 130, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 199, 252, 252, 253, 252, 252, 233, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 128, 252, 253, 252, 141, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    scores = estimator.predict(input_fn=make_online_prediction_input_fn(instances))
    for x in scores:
        print(x)

    print('Done.')

def main():
    model_dir = '/tmp/mnist/model'

    #train(model_dir)
    eval(model_dir)
    #batch_predict(model_dir)
    for i in range(10):
        online_predict(model_dir)


if __name__ == "__main__":
    main()
