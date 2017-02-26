import tensorflow as tf


def read_tensors_from_csv(file_name, defaults=None, num_columns=None, batch_size=1, num_epochs=None,
                          delimiter=',', randomize_input=True, num_threads=4):
    if file_name is None:
        raise ValueError(
            "Invalid file_name. file_name cannot be empty.")

    if defaults is None and num_columns is None:
        raise ValueError(
            "At least one of defaults and num_columns should not be None.")

    if defaults is None:
        defaults = [0.0 for _ in range(num_columns)]

    record_defaults = [[item] for item in defaults]

    examples = tf.contrib.learn.read_batch_examples(
        file_pattern=file_name,
        batch_size=batch_size,
        reader=tf.TextLineReader,
        randomize_input=randomize_input,
        num_threads=num_threads,
        num_epochs=num_epochs)

    columns = tf.decode_csv(
        examples, record_defaults=record_defaults, field_delim=delimiter)

    return columns
