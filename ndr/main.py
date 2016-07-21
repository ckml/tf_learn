from lib import *
import matplotlib.pyplot as plt
import numpy as np
from cnn import *


def PreprareData(folder):
    labels, features = LoadData(folder + "/train")
    print len(labels), len(features)

    labels_file = folder + "/train.labels"
    np.save(labels_file, labels)

    features_file = folder + "/train_raw.features"
    np.save(features_file, features)

    features = NormalizeImages(features)
    features_file = folder + "/train.features"
    np.save(features_file, features)

    labels, features = LoadData(folder + "/test")
    print len(labels), len(features)

    labels_file = folder + "/test.labels"
    np.save(labels_file, labels)

    features_file = folder + "/test_raw.features"
    np.save(features_file, features)

    features = NormalizeImages(features)
    features_file = folder + "/test.features"
    np.save(features_file, features)


def main():
    folder = "/tmp"

    PreprareData(folder)

    train_features = np.load(folder + "/train.features.npy")
    train_labels = np.load(folder + "/train.labels.npy")
    train_features_raw = np.load(
        folder + "/train_raw.features.npy")

    test_features = np.load(folder + "/test.features.npy")
    test_labels = np.load(folder + "/test.labels.npy")
    test_features_raw = np.load(
        folder + "/test_raw.features.npy")

    train_data = {'train_features': train_features,
                  'train_labels': train_labels,
                  'valid_features': test_features, 'valid_labels': test_labels}

    check_point_file = folder + '/model.ckpt'

    model = CreateModel()
    Train(model, train_data, 100, 100, check_point_file)

    test_data = {'features': test_features, 'labels': test_labels,
                 'raw_features': test_features_raw}

    batch_prediction = BatchPredict(model, test_data, check_point_file)

    count = 0
    for i in range(0, test_data['labels'].shape[0]):
        predicted_label = np.argmax(batch_prediction[i, :])
        if test_data['labels'][i, predicted_label] != 1:
            count += 1
            print count, "Truth: ", np.argmax(
                test_data['labels'][i, :]), "     Predicted: ", \
                predicted_label, "     Score: ", batch_prediction[
                i, predicted_label]
            img = np.zeros([28, 28 * 2])
            tmp = test_data['features'][i, :]
            tmp.shape = (28, 28)
            img[:, 0:28] = tmp
            tmp = test_data['raw_features'][i, :]
            tmp /= np.max(tmp)
            tmp.shape = (28, 28)
            img[:, 28:56] = tmp
            plt.imshow(img)
            plt.show()

    print 'Accuracy: ', 1.0 - 1.0 * count / test_data['labels'].shape[0]


if __name__ == "__main__":
    main()
