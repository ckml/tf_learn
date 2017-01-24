import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import to_categorical
import pandas as pd

from sklearn import cross_validation
from sklearn import metrics

columns = ['timestamp', 'activityid', 'heartrate', 'handtemperature',
           'handacceleration1', 'handacceleration2', 'handacceleration3',
           'handacceleration4', 'handacceleration5', 'handacceleration6',
           'handgyroscope1', 'handgyroscope2', 'handgyroscope3',
           'handmagnetometer1', 'handmagnetometer2', 'handmagnetometer3',
           'handinvalid1', 'handinvalid2', 'handinvalid3', 'handinvalid4',
           'chesttemperature', 'chestacceleration1', 'chestacceleration2',
           'chestacceleration3', 'chestacceleration4', 'chestacceleration5',
           'chestacceleration6', 'chestgyroscope1', 'chestgyroscope2',
           'chestgyroscope3', 'chestmagnetometer1', 'chestmagnetometer2',
           'chestmagnetometer3', 'chestinvalid1', 'chestinvalid2',
           'chestinvalid3', 'chestinvalid4', 'ankletemperature',
           'ankleacceleration1', 'ankleacceleration2', 'ankleacceleration3',
           'ankleacceleration4', 'ankleacceleration5', 'ankleacceleration6',
           'anklegyroscope1', 'anklegyroscope2', 'anklegyroscope3',
           'anklemagnetometer1', 'anklemagnetometer2', 'anklemagnetometer3',
           'ankleinvalid1', 'ankleinvalid2', 'ankleinvalid3', 'ankleinvalid4']

featurescolumns = ['heartrate', 'handtemperature', 'handacceleration1',
                   'handacceleration2', 'handacceleration3', 'handgyroscope1',
                   'handgyroscope2', 'handgyroscope3', 'handmagnetometer1',
                   'handmagnetometer2', 'handmagnetometer3', 'chesttemperature',
                   'chestacceleration1', 'chestacceleration2',
                   'chestacceleration3', 'chestgyroscope1', 'chestgyroscope2',
                   'chestgyroscope3', 'chestmagnetometer1',
                   'chestmagnetometer2', 'chestmagnetometer3',
                   'ankletemperature', 'ankleacceleration1',
                   'ankleacceleration2', 'ankleacceleration3',
                   'anklegyroscope1', 'anklegyroscope2', 'anklegyroscope3',
                   'anklemagnetometer1', 'anklemagnetometer2',
                   'anklemagnetometer3']

LABEL_COLUMN = 'activityid'


def preparefiles():
    filenames = ['subject101.dat', 'subject102.dat', 'subject103.dat',
                 'subject104.dat', 'subject105.dat', 'subject106.dat',
                 'subject107.dat', 'subject108.dat', 'subject109.dat']
    with open('merged.dat', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    df = pd.read_csv('merged.dat', skipinitialspace=True, na_values='NaN',
                     sep=' ', header=None, names=columns)
    # drop rows with nan value
    df = df.dropna()

    # drop rows with activityid as 0
    df = df[df.activityid != 0]
    dfn = df.copy()

    for col in df.columns:
        if col in LABEL_COLUMN:
            continue

        meanV = df[col].mean()
        stdV = df[col].std()
        dfn[col] = (df[col] - meanV) / stdV

    dfn.to_csv("normalized.csv", index=False)
    # split into train and test
    msk = np.random.rand(len(df)) < 0.7
    train = dfn[msk]
    test = dfn[~msk]
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
    return (100.0 * np.sum(pred_class == true_class) / len(predictions))


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration


# preparefiles()
train_file_name = 'train.csv'
test_file_name = 'test.csv'
df_train = pd.read_csv(
    train_file_name,
    skipinitialspace=True)
df_test = pd.read_csv(
    test_file_name,
    skipinitialspace=True)

train_features = df_train[featurescolumns].values
X = train_features.astype(np.float32, copy=False)
train_labels = df_train[LABEL_COLUMN].values
train_labels = train_labels.astype(np.int32, copy=False)
train_labels = train_labels - 1
Y = to_categorical(train_labels, 24)

test_features = df_test[featurescolumns].values
X_val = test_features.astype(np.float32, copy=False)
test_labels = df_test[LABEL_COLUMN].values
test_labels = test_labels.astype(np.int32, copy=False)
test_labels = test_labels - 1
Y_val = to_categorical(test_labels, 24)

tflearn.init_graph(num_cores=15)
net = tflearn.input_data([None, 31])
net = tflearn.fully_connected(net, 100, activation='relu',
                              weights_init='xavier')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 300, activation='relu',
                              weights_init='xavier')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 500, activation='relu',
                              weights_init='xavier')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 300, activation='relu',
                              weights_init='xavier')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 100, activation='relu',
                              weights_init='xavier')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 24, activation='softmax')
#  sgd = tflearn.SGD(learning_rate=1.0, lr_decay=0.96, decay_step=500)
#  opt = tflearn.optimizers.AdaDelta(learning_rate=0.001, rho=0.1, epsilon=1e-08, use_locking=False, name='AdaDelta')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
model = tflearn.DNN(net, best_val_accuracy=0.8, tensorboard_dir='tflearn_logs',
                    checkpoint_path="check_point",
                    best_checkpoint_path="best_checkpoint_path")

# Initializae our callback.
early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.98)

model.fit(X, Y, show_metric=True, batch_size=len(X), n_epoch=10000,
          snapshot_epoch=True,
          snapshot_step=1000, validation_set=(X_val, Y_val),
          callbacks=early_stopping_cb)

model.save("model.tfl")

prediction_val = model.predict(X_val)
prediction = model.predict(X)
print 'Training on'
print 'Validation accuracy: %.1f%%' % accuracy(prediction_val, Y_val)
print 'Test accuracy: %.1f%%' % accuracy(prediction, Y)