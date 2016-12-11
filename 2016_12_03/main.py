from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
# from six.moves import urllib

import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/tmp/poker_hand", "Base directory for output models.")
flags.DEFINE_string("model_type", "deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 2000, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

# 1,10,1,11,1,13,1,12,1,1,9
COLUMNS = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "Result"]
LABEL_COLUMN = "label"
CONTINUOUS_COLUMNS = []
CATEGORICAL_COLUMNS = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5",
                       "C5"]


def build_estimator(model_dir):
    """Build an estimator."""
    # Continuous base columns.
    S1 = tf.contrib.layers.sparse_column_with_integerized_feature("S1", 4)
    C1 = tf.contrib.layers.sparse_column_with_integerized_feature("C1", 13)
    S2 = tf.contrib.layers.sparse_column_with_integerized_feature("S2", 4)
    C2 = tf.contrib.layers.sparse_column_with_integerized_feature("C2", 13)
    S3 = tf.contrib.layers.sparse_column_with_integerized_feature("S3", 4)
    C3 = tf.contrib.layers.sparse_column_with_integerized_feature("C3", 13)
    S4 = tf.contrib.layers.sparse_column_with_integerized_feature("S4", 4)
    C4 = tf.contrib.layers.sparse_column_with_integerized_feature("C4", 13)
    S5 = tf.contrib.layers.sparse_column_with_integerized_feature("S5", 4)
    C5 = tf.contrib.layers.sparse_column_with_integerized_feature("C5", 13)

    # Wide columns and deep columns.
    deep_columns = [tf.contrib.layers.embedding_column(S1, dimension=4),
                    tf.contrib.layers.embedding_column(C1, dimension=13),
                    tf.contrib.layers.embedding_column(S2, dimension=4),
                    tf.contrib.layers.embedding_column(C2, dimension=13),
                    tf.contrib.layers.embedding_column(S3, dimension=4),
                    tf.contrib.layers.embedding_column(C3, dimension=13),
                    tf.contrib.layers.embedding_column(S4, dimension=4),
                    tf.contrib.layers.embedding_column(C4, dimension=13),
                    tf.contrib.layers.embedding_column(S5, dimension=4),
                    tf.contrib.layers.embedding_column(C5, dimension=13),
                    ]

    model = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[50, 100, 20],
                                           n_classes=10)
    return model


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval():
    """Train and evaluate the model."""
    train_file_name = "/tmp/poker-hand-training-true.data"
    test_file_name = train_file_name

    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=False,
        skiprows=1,
        engine="python")
    print(df_train['S1'].size)

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    df_train[LABEL_COLUMN] = (
        df_train["Result"].apply(lambda x: x)).astype(int)
    df_test[LABEL_COLUMN] = (
        df_test["Result"].apply(lambda x: x)).astype(int)

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir)
    m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


train_and_eval()
