import tensorflow as tf


def read_tensors_from_csv(file_name, header=None, defaults=None, num_columns=None, batch_size=1, num_epochs=None,
                          delimiter=',', randomize_input=True, num_threads=4):
    if file_name is None:
        raise ValueError(
            "Invalid file_name. file_name cannot be empty.")

    if header is None and defaults is None and num_columns is None:
        raise ValueError(
            "At least one of header, defaults and num_columns should not be None.")

    if header:
        num_columns = len(header)
    elif defaults:
        num_columns = len(defaults)

    if header is None:
        header = [str(i) for i in range(num_columns)]
    elif len(header) != num_columns:
        raise ValueError(
            "Invalid header length.")

    if defaults is None:
        defaults = [0.0 for _ in range(num_columns)]
    elif len(defaults) != num_columns:
        raise ValueError(
            "Invalid defaults length.")

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
    features = dict(zip(header, columns))

    return features
