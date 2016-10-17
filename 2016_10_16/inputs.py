"""Input handling for UCI Census dataset.
"""

import math
import sys
import tensorflow as tf
from tensorflow.python.lib.io import file_io

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 3,
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
NUMERIC_FEATURES = ['age', 'education-num', 'hours-per-week', 'capital-gain',
                    'capital-loss']
CATEGORICAL_FEATURES = ['marital-status', 'native-country', 'workclass', 'sex',
                        'education', 'relationship', 'occupation', 'race']
OUT_OF_VOCAB_KEY = 'UNKNOWN'
MAX_READ_THREADS = 2  # On Haswell machines, there are 36 physical cores.


def _files(pattern):
    """Converts a file pattern to a list of files."""
    files = file_io.get_matching_files(pattern)
    if not files:
        raise IOError('Unable to find input files.')
    return files


def _file_lines(text_file):
    """Read a text file and returns its lines in a list."""
    return file_io.read_file_to_string(text_file).split()


def _feature_spec():
    """Defines the input features in each example."""
    features = {LABEL: tf.FixedLenFeature(
        shape=[], dtype=tf.float32, default_value=0.0)}
    for i in NUMERIC_FEATURES:
        features[i] = tf.FixedLenFeature(
            shape=[], dtype=tf.float32, default_value=0.0)
    for c in CATEGORICAL_FEATURES:
        features[c] = tf.FixedLenFeature(
            shape=[], dtype=tf.string, default_value='')
    return features


def _read(filename_queue):
    """Reads serialized examples."""
    reader = tf.TextLineReader()

    _, serialized_examples = reader.read_up_to(filename_queue, FLAGS.batch_size)
    return serialized_examples,


# def _embed(features):
#     """Embedding lookup for categorical features."""
#     embeddings = []
#     with tf.name_scope('embed'):
#         for c in CATEGORICAL_FEATURES:
#             vocab_file = FLAGS.vocab_dir + '/' + c + '.txt'
#             # print(vocab_file)
#             vocab = _file_lines(vocab_file)
#             vocab_size = len(vocab) + 1
#             embedding_size = int(math.floor(6 * vocab_size ** 0.25))
#             # print(c, embedding_size)
#             embedding_weights = tf.Variable(
#                 tf.truncated_normal(
#                     [vocab_size, embedding_size],
#                     stddev=1.0 / math.sqrt(vocab_size)),
#                 name='%s_embedding' % c)
#             ids = tf.contrib.lookup.string_to_index(
#                 features[c],
#                 mapping=[OUT_OF_VOCAB_KEY] + vocab,
#                 default_value=0,
#                 name='%s_lookup' % c)
#             embedding = tf.nn.embedding_lookup(
#                 embedding_weights, ids, name='%s_lookup_embedding' % c)
#             embeddings.append(embedding)
#     return embeddings


def _transform(data):
    """Transform features into examples and labels."""
    labels = data.pop(LABEL)
    features = [data["age"], data["fnlwgt"], data["education-num"],
                data["capital-gain"], data["capital-loss"],
                data["hours-per-week"]]

    #TODO: handle data["workclass"], data["fnlwgt"], data["education"], data[
    #    "marital-status"], data["occupation"], data["relationship"], data[
    #    "race"], data["sex"], data["native-country"]

    return features, labels


def _decode(example_batch):
    """Decode a batch of CSV lines into a feature map."""

    if FLAGS.is_predicting:
        record_defaults = [[0.0], [""], [0.0], [""], [0], [""], [""], [""],
                           [""],
                           [""], [0], [0], [0], [""]]
    else:
        record_defaults = [[0.0], [""], [0.0], [""], [0], [""], [""], [""],
                           [""], [""], [0], [0], [0], [""], [""]]

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
            files, num_epochs=epochs, shuffle=is_training,
            name='filename_queue')

        read_threads = min(len(files), MAX_READ_THREADS)
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
