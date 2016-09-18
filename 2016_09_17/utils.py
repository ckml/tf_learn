import copy as cp
import csv
import matplotlib.pyplot as plt
import numpy as np


def Repeat(item, n):
    data = []
    for i in range(n):
        data.append(cp.deepcopy(item))
    return data


def ReadCSV(file, delimiter=','):
    cols = 0
    row_index = 0
    with open(file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='\"')
        for row in reader:
            row_index = row_index + 1
            if cols == 0 and len(row) > 0:
                cols = len(row)
                data = Repeat([], cols)
            elif cols != len(row):
                print 'Skipped invalid row: ', row_index, len(row), '/', cols
                continue
            for i in range(len(row)):
                data[i].append(row[i])
    return data


def CreateLabels(string_labels, one_of_n=True):
    label_map = {}
    labels = []
    if one_of_n:
        for string_label in string_labels:
            if not label_map.has_key(string_label):
                index = len(label_map)
                label_map[string_label] = index
        num_classes = len(label_map)
        for string_label in string_labels:
            label = np.zeros(num_classes)
            label[label_map.get(string_label)] = 1.0
            labels.append(label)
    else:
        for string_label in string_labels:
            if not label_map.has_key(string_label):
                index = len(label_map)
                label_map[string_label] = index
            labels.append(label_map.get(string_label))
    labels = np.array(labels)
    return labels, label_map


def ReadData(file, delimiter=',', one_of_n=True):
    data = ReadCSV(file, delimiter)
    num_cols = len(data)
    labels, label_map = CreateLabels(data[num_cols-1], one_of_n)
    features = np.array(data[0:num_cols-1]).astype(np.float)
    features = np.transpose(features)
    data = {'features': features, 'labels': labels, 'label_map': label_map}
    return data


def SplitData(data, training_data_ratio=0.9):
    num_instances = data['features'].shape[0]
    indices = np.arange(num_instances)
    np.random.shuffle(indices)

    pivot = int(num_instances * training_data_ratio)

    training_features = data['features'][indices[0:pivot], :]
    training_labels = data['labels'][indices[0:pivot], :]

    test_features = data['features'][indices[pivot:num_instances], :]
    test_labels = data['labels'][indices[pivot:num_instances], :]

    training_data = {'features': training_features, 'labels': training_labels}
    test_data = {'features': test_features, 'labels': test_labels}

    return training_data, test_data

def NormalizeFeatures(features):
    num_features = features.shape[1]
    max_features = np.empty([num_features])
    for feature in features:
        for i in np.arange(num_features):
            if max_features[i] < np.abs(feature[i]):
                max_features[i] = np.abs(feature[i])
    max_features += 1e-20
    for feature in features:
        for i in np.arange(num_features):
            feature[i] = feature[i] / max_features[i]
    return features

def CrossFeatures(features):
    num_instances = features.shape[0]
    num_features = features.shape[1]

    updated_num_features = (num_features + 1) * num_features / 2
    updated_features = np.empty(
        [num_instances, num_features + updated_num_features])
    for i in np.arange(num_instances):
        f = features[i, :]
        updated_features[i, 0:num_features] = f
        index = num_features
        for j in np.arange(num_features):
            for k in np.arange(j, num_features):
                updated_features[i, index] = f[j] * f[k]
                index = index + 1
    return updated_features

def RandomProject(features, target_num_features):
    num_instances = features.shape[0]
    num_features = features.shape[1]

    updated_features = np.empty([num_instances, target_num_features])
    rand_proj_mat = np.random.randn(num_features, target_num_features)
    for i in np.arange(num_instances):
        f = features[i, :]
        updated_features[i, :] = np.matmul(f, rand_proj_mat)
    return updated_features

def ExtractKernelFeatures(features):
    return features