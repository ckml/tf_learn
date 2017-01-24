from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tensorflow.python.platform import gfile
import csv

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# DATA SET
FOLDER = 'd:\\AI\\Project\\uci\\PAMAP2\\PAMAP2_Dataset\\Protocol\\'
FILE1 = 'subject101.dat' #376,417 rows
MODEL_FOLDER = 'd:\\AI\\Project\\uci\\model'

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            col_delimiter =' ',
                            target_column=-1):

    """Load dataset from CSV file without a header row."""
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file, delimiter = col_delimiter)
        data, target = [], []
        for row in data_file:
            target_v = int(row.pop(target_column))
            if target_v == 0:
                continue
            elif target_v < 8:
                target_v = target_v - 1
            elif target_v < 14:
                target_v = target_v - 2
            elif target_v < 21:
                target_v = target_v - 4
            elif target_v == 24:
                target_v = 17
            target.append(target_v) #row.pop(target_column)
            row = row[1:53]
            row = np.asarray(row, dtype=features_dtype)
            row = np.nan_to_num(row)
            data.append(row)
    target = np.array(target, dtype=target_dtype)
    data = np.array(data)
    return Dataset(data=data, target=target)

#Load datasets
def load_datasets(cxv_file):
    csv_data = load_csv_without_header(
        filename = cxv_file,
        target_dtype = np.int,
        features_dtype = np.float32,
        col_delimiter=' ',
        target_column = 1) # 2nd column is labels

    return csv_data

csv_data = load_datasets(FOLDER+FILE1)
csv_rows = csv_data.target.size - 1
csv_num_feature_columns = len(csv_data[0])

# Specify that all features have real-value data, also has SparseColumn, BucketizedColumn(real -> bucket), CrossedColumn
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=csv_num_feature_columns)]
# feature crossing: bucketized_column then cross_column - sport_x_city = tf.contrib.layers.crossed_column( [sport, city], hash_bucket_size=int(1e4))


idx_training = np.random.randint(csv_rows, size=int(csv_rows*0.7))
train_set = Dataset(data=csv_data.data[idx_training], target=csv_data.target[idx_training])
test_set = Dataset(data=csv_data.data[~idx_training], target=csv_data.target[~idx_training])

#validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(test_set.data,test_set.target,every_n_steps=50)
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[80, 160, 80],
                                            n_classes=18,
                                            model_dir=MODEL_FOLDER,
                                            optimizer=tf.train.FtrlOptimizer(
                                                learning_rate=0.05,
                                                l1_regularization_strength=0.1
                                            ),
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=2),
                                            dropout=0.5
                                            )#

# Fit model.
classifier.fit(x=train_set.data,
               y=train_set.target,
               steps=4000,
               batch_size=1000,
               monitors=[validation_monitor])


# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
#tensorboard --logdir=d:\AI\Project\uci\\model



