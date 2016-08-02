from utils import *
from lib import *


def Test01():
    # The data file can be downloaded from
    #     https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
    data = ReadData("/tmp/Skin_NonSkin.txt", '\t')

    #data['features'] = NormalizeFeatures(data['features'])
    #data['features'] = CrossFeatures(data['features'])

    print 'num_instances:', data['features'].shape[0], 'num_features:', \
        data['features'].shape[1], 'num_classes:', data['labels'].shape[1]

    train_data, test_data = SplitData(data, 0.8)

    num_features = data['features'].shape[1]
    num_classes = len(data['label_map'])

    model = CreateLM(num_features, num_classes, learning_rate=0.001)
    Train(model, train_data, test_data, 100, 1000, stopping_threshold=0.2,
          check_point_file='/tmp/model.ckpt', resume_training=False)


Test01()
