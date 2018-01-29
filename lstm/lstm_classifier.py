import os

from absl import flags
import tensorflow as tf


flags.DEFINE_string(
    "train_pattern",
    "/tmp/data/train_data",
    "Filepattern for a TFRecordio of training data.")

flags.DEFINE_string(
    "eval_pattern",
    "/tmp/data/test_data",
    "Filepattern for an TFRecordio of evaluation data.")

flags.DEFINE_float(
    "learning_rate", 0.1,
    "Learning rate - see go/lower for more details about tuning this.")

flags.DEFINE_string(
    "optimizer_type", "Adagrad",
    "Optimizer type. Should be one of 'Adagrad', 'Momentum', or 'SGD'.")

flags.DEFINE_string(
    "cell_type", "lstm",
    "The RNN cell type to use. Must be one of 'basic_rnn', 'lstm', or 'gru'.")

flags.DEFINE_list(
    "rnn_cell_num_units", ["32"],
    "A list containing the size of the RNN cell in each layer.")

flags.DEFINE_list(
    "dropout_keep_probabilities", ["1.0", "1.0"],
    "Dropout keep probabilities. Must have length num_layers + 1.")

flags.DEFINE_integer(
    "embedding_hash_buckets", 1000,
    "Number of buckets to hash review terms into before embedding.")

flags.DEFINE_integer(
    "embedding_dimension", 32,
    "Dimension of the word embedding.")

flags.DEFINE_integer(
    "batch_size", 16, "Batch size.")

flags.DEFINE_integer(
    "num_parsing_threads", 4,
    "The number of threads used to parse Examples.")

flags.DEFINE_integer(
    "input_queue_capacity", 128,
    "The size of the queue of parsed .")

flags.DEFINE_list(
    "sequence_bucketing_boundaries", ["100", "200", "300", "400"],
    "If non-empty, these are the boundaries used to bucket and batch input "
    "sequences by length.")

flags.DEFINE_integer(
    "num_train_steps", None,
    "Number of training iterations. None means continuous training.")

flags.DEFINE_integer(
    "num_eval_steps", 500,
    "Number of evaluation iterations. When running continuous_eval, this is "
    "the number of eval steps run for each evaluation of a checkpoint.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "Interval at which to save checkpoints.")

FLAGS = flags.FLAGS

EXAMPLES_KEY = "examples"
SEQUENCE_LENGTH_KEY = "sequence_length"
LABEL_KEY = "labels"
TERMS_KEY = "terms"


def sparse_sequence_length(sparse_tensor):
  with tf.name_scope("sparse_sequence_length"):
    indices = tf.to_int32(sparse_tensor.indices)
    row_indices = indices[:, 0]
    col_indices = indices[:, 1]
    num_rows = tf.to_int32(sparse_tensor.dense_shape[0])
    row_range = tf.expand_dims(tf.range(num_rows), 0)
    row_indicator = tf.to_int32(
        tf.equal(tf.expand_dims(row_indices, 1), row_range))
    split_col_indices = row_indicator * (tf.expand_dims(col_indices, 1) + 1)
    row_lengths = tf.reduce_max(split_col_indices, [0])
  return row_lengths


def _lookup_probabilities(predictions, probabilities):
  predictions = tf.cast(predictions, tf.int32)
  rang = tf.range(tf.shape(predictions)[0])
  indices = tf.concat(
      [tf.expand_dims(rang, 1), tf.expand_dims(predictions, 1)], 1)
  prediction_probabilities = tf.gather_nd(probabilities, indices)
  return prediction_probabilities


def get_sequence_feature_columns(num_hash_buckets, embedding_dimension):
  terms = tf.feature_column.sparse_column_with_hash_bucket(
      TERMS_KEY, hash_bucket_size=num_hash_buckets, combiner="sum")
  embedded_terms = tf.feature_column.embedding_column(
      terms, embedding_dimension, combiner="sum")
  return (embedded_terms,)


def get_input_fn(file_pattern,
                 batch_size,
                 sequence_bucketing_boundaries,
                 num_threads,
                 queue_capacity,
                 name):
  file_names = os.glob.Glob(file_pattern)
  def input_fn():
    with tf.name_scope(name):
      file_queue = tf.train.string_input_producer(file_names)
      reader = tf.TFRecordReader()
      _, serialized = reader.read(file_queue)
      parsing_spec = {
          LABEL_KEY: tf.FixedLenFeature(
              shape=[1], dtype=tf.float32, default_value=None),
          TERMS_KEY: tf.VarLenFeature(dtype=tf.string)}
      features = tf.parse_single_example(serialized, parsing_spec)

      # Labels are required to be `int32`.
      features[LABEL_KEY] = tf.to_int32(features[LABEL_KEY])

      # Sequence length is needed for masking out padding.
      sequence_length = tf.shape(features[TERMS_KEY])[0]
      tf.contrib.deprecated.histogram_summary(
          "{}/sequence_length".format(name), sequence_length)
      features[SEQUENCE_LENGTH_KEY] = sequence_length

      if sequence_bucketing_boundaries:
        # Batch using buckets defined by `sequence_bucketing_boundaries.
        _, batched_features = tf.contrib.training.bucket_by_sequence_length(
            input_length=sequence_length,
            tensors=features,
            bucket_boundaries=sequence_bucketing_boundaries,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=queue_capacity,
            dynamic_pad=True)
      else:
        # Use a conventional batching operation with no bucketing.
        batched_features = tf.batch(
            tensors=features,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=queue_capacity,
            enqueue_many=False,
            dynamic_pad=True)
    label = batched_features.pop(LABEL_KEY)
    return batched_features, label
  return input_fn


def serving_input_fn():
  with tf.name_scope("inputs"):
    serialized = tf.placeholder(
        dtype=tf.string,
        shape=tf.tensor_shape.unknown_shape(ndims=1),
        name=EXAMPLES_KEY)

    parsing_spec = {TERMS_KEY: tf.VarLenFeature(dtype=tf.string)}
    features = tf.parse_example(serialized, parsing_spec)

    sequence_length = sparse_sequence_length(features[TERMS_KEY])
    features[SEQUENCE_LENGTH_KEY] = sequence_length
    return tf.contrib.learn.InputFnOps(
        features=features,
        labels=None,
        default_inputs={EXAMPLES_KEY: serialized})


def _experiment_fn(output_dir):
  sequence_feature_columns = get_sequence_feature_columns(
      FLAGS.embedding_hash_buckets, FLAGS.embedding_dimension)

  sequence_bucketing_boundaries = [
      int(b) for b in FLAGS.sequence_bucketing_boundaries
  ]

  train_input_fn = get_input_fn(FLAGS.train_pattern,
                                FLAGS.batch_size,
                                sequence_bucketing_boundaries,
                                FLAGS.num_parsing_threads,
                                FLAGS.input_queue_capacity,
                                name="training_input")

  eval_input_fn = get_input_fn(FLAGS.eval_pattern,
                               FLAGS.batch_size,
                               sequence_bucketing_boundaries,
                               FLAGS.num_parsing_threads,
                               FLAGS.input_queue_capacity,
                               name="eval_input")

  config = tf.contrib.learn.learn_runner.EstimatorConfig(
      save_checkpoints_secs=None,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps)

  dropout_keep_probabilities = [
      float(d) for d in FLAGS.dropout_keep_probabilities]

  rnn_cell_num_units = [int(n) for n in FLAGS.rnn_cell_num_units]

  export_strategy = tf.contrib.learn.make_export_strategy(serving_input_fn)

  movie_review_classifier = tf.contrib.learn.DynamicRnnEstimator(
      problem_type=tf.contrib.learn.ProblemType.CLASSIFICATION,
      prediction_type=1, # PredictionType.SINGLE_VALUE,
      sequence_feature_columns=sequence_feature_columns,
      num_classes=2,
      num_units=rnn_cell_num_units,
      cell_type=FLAGS.cell_type,
      optimizer=FLAGS.optimizer_type,
      learning_rate=FLAGS.learning_rate,
      predict_probabilities=True,
      momentum=FLAGS.momentum,
      dropout_keep_probabilities=dropout_keep_probabilities,
      model_dir=output_dir,
      config=config)


  movie_review_experiment = tf.contrib.learn.Experiment(
      estimator=movie_review_classifier,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=FLAGS.num_train_steps,
      eval_steps=FLAGS.num_eval_steps,
      eval_delay_secs=0,
      continuous_eval_throttle_secs=5,
      export_strategies=[export_strategy])

  return movie_review_experiment


def main(unused_argv):
  run_config = tf.contrib.learn.RunConfig()
  run_config = run_config.replace(model_dir="/tmp/lstm10")
  tf.contrib.learn.learn_runner.run(experiment_fn=_experiment_fn,
                   run_config=run_config,  # RunConfig
                   schedule="train_and_evaluate")


if __name__ == "__main__":
  tf.app.run()
