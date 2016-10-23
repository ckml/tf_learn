"""Input handling for UCI Census dataset.
"""

import math
import sys
import tensorflow as tf
from tensorflow.python.lib.io import file_io

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 5,
                     'Number of input records used per batch.')
flags.DEFINE_boolean(
    'is_predicting', False,
    'Whether we are doing prediciton. This is a hack to get reading TSV test '
    'file working.')
flags.DEFINE_integer(
    'input_queue_memory_factor', 16,
    'Size of the queue of serialized examples. Default is ideal but try '
    'smaller values, e.g. 4, 2 or 1, if host memory is constrained.')

LABEL = 'over_50k'
OUT_OF_VOCAB_KEY = 'UNKNOWN'
MAX_READ_THREADS = 1  # On Haswell machines, there are 36 physical cores.

VOCAB = {}
VOCAB[LABEL] = [" <=50K", " >50K"]
VOCAB["workclass"] = [" Private", " Self-emp-not-inc", " Self-emp-inc",
                      " Federal-gov", " Local-gov", " State-gov",
                      " Without-pay", " Never-worked"]
VOCAB["education"] = [" Bachelors", " Some-college", " 11th", " HS-grad",
                      " Prof-school", " Assoc-acdm", " Assoc-voc", " 9th",
                      " 7th-8th", " 12th", " Masters", " 1st-4th", " 10th",
                      " Doctorate", " 5th-6th", " Preschool"]
VOCAB["marital-status"] = [" Married-civ-spouse", " Divorced",
                           " Never-married", " Separated", " Widowed",
                           " Married-spouse-absent", " Married-AF-spouse"]
VOCAB["occupation"] = [" Tech-support", " Craft-repair", " Other-service",
                       " Sales", " Exec-managerial", " Prof-specialty",
                       " Handlers-cleaners", " Machine-op-inspct",
                       " Adm-clerical", " Farming-fishing",
                       " Transport-moving", " Priv-house-serv",
                       " Protective-serv", " Armed-Forces"]
VOCAB["relationship"] = [" Wife", " Own-child", " Husband",
                         " Not-in-family",
                         " Other-relative", " Unmarried"]
VOCAB["race"] = [" White", " Asian-Pac-Islander", " Amer-Indian-Eskimo",
                 " Other", " Black"]
VOCAB["sex"] = [" Female", " Male"]
VOCAB["native-country"] = [" United-States", " Cambodia", " England",
                           " Puerto-Rico", " Canada", " Germany",
                           " Outlying-US(Guam-USVI-etc)", " India",
                           " Japan",
                           " Greece", " South", " China", " Cuba", " Iran",
                           " Honduras", " Philippines", " Italy", " Poland",
                           " Jamaica", " Vietnam", " Mexico", " Portugal",
                           " Ireland", " France", " Dominican-Republic",
                           " Laos", " Ecuador", " Taiwan", " Haiti",
                           " Columbia",
                           " Hungary", " Guatemala", " Nicaragua",
                           " Scotland",
                           " Thailand", " Yugoslavia", " El-Salvador",
                           " Trinadad&Tobago", " Peru", " Hong",
                           " Holand-Netherlands"]


def _files(pattern):
    """Converts a file pattern to a list of files."""
    files = file_io.get_matching_files(pattern)
    if not files:
        raise IOError('Unable to find input files.')
    return files

def _read(filename_queue):
    """Reads serialized examples."""
    reader = tf.TextLineReader()

    _, serialized_examples = reader.read_up_to(filename_queue, FLAGS.batch_size)
    return serialized_examples,


def _transform(data):
    """Transform features into examples and labels."""

    labels = tf.contrib.lookup.string_to_index(
        data.pop(LABEL),
        mapping=VOCAB[LABEL],
        default_value=0,
        name=LABEL)

    labels = tf.one_hot(labels, 2, axis=-1)

    columns = [data.pop("age"), data.pop("fnlwgt"), data.pop("education-num"),
                data.pop("capital-gain"), data.pop("capital-loss"),
                data.pop("hours-per-week")]

    num_features = []
    for column in columns:
        column = tf.transpose(column)
        num_features.append(column)
    num_features = tf.concat(0, [num_features])
    num_features = tf.transpose(num_features)

    features = []
    features.append(num_features)
    for feature in data:
        indices = tf.contrib.lookup.string_to_index(
            data[feature],
            mapping=[OUT_OF_VOCAB_KEY] + VOCAB[feature],
            default_value=0,
            name='%s_lookup' % feature)
        one_hot_vec = tf.one_hot(indices, len(VOCAB[feature]) + 1, on_value=1.0,
                                 axis=-1)
        features.append(one_hot_vec)
    features = tf.concat(1, features)

    return features, labels


def _decode(example_batch):
    """Decode a batch of CSV lines into a feature map."""

    if FLAGS.is_predicting:
        record_defaults = [[0.0], [""], [0.0], [""], [0.0], [""], [""], [""],
                           [""], [""], [0.0], [0.0], [0.0], [""]]
    else:
        record_defaults = [[0.0], [""], [0.0], [""], [0.0], [""], [""], [""],
                           [""], [""], [0.0], [0.0], [0.0], [""], [""]]

    fields = tf.decode_csv(example_batch, record_defaults, field_delim=',')
    if FLAGS.is_predicting:
        data = {LABEL: tf.constant("")}
    else:
        data = {LABEL: fields[14]}

    data["age"] = fields[0]
    data["workclass"] = fields[1]
    data["fnlwgt"] = fields[2]
    data["education"] = fields[3]
    data["education-num"] = fields[4]
    data["marital-status"] = fields[5]
    data["occupation"] = fields[6]
    data["relationship"] = fields[7]
    data["race"] = fields[8]
    data["sex"] = fields[9]
    data["capital-gain"] = fields[10]
    data["capital-loss"] = fields[11]
    data["hours-per-week"] = fields[12]
    data["native-country"] = fields[13]

    return data


def inputs(pattern, is_training=True):
    """Construct batches of training or evaluation examples from the input files.

    Args:
      pattern: String. Pattern specifying the input files.
      is_training: Boolean. Whether or not the model is in training mode.

    Returns:
      examples: 2-D float Tensor of [batch_size, example_length].
      labels: 1-D float Tensor of [batch_size].
    """
    with tf.name_scope('input'):
        files = _files(pattern)
        if is_training:
            epochs = sys.maxint  # Training is controlled by steps.
        else:
            epochs = 1  # Evaluation should go through input data exactly once.
        filename_queue = tf.train.string_input_producer(
            files, num_epochs=epochs, shuffle=False,  # is_training,
            name='filename_queue')

        # read_threads = min(len(files), MAX_READ_THREADS)
        read_threads = 1
        examples_list = [_read(filename_queue) for _ in range(read_threads)]
        min_after_dequeue = 1024 * FLAGS.input_queue_memory_factor
        capacity = min_after_dequeue + read_threads * FLAGS.batch_size
        if is_training:
            example_batch = tf.train.shuffle_batch_join(
                examples_list,
                batch_size=FLAGS.batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                enqueue_many=True)
        else:
            example_batch = tf.train.batch_join(
                examples_list,
                batch_size=FLAGS.batch_size,
                capacity=capacity,
                enqueue_many=True)

        data = _decode(example_batch)
        return _transform(data)
