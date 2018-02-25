"""
 Functions to load images and their labels,
 feature vectors from the sample folder

 Date: 18th February, 18
"""

import warnings

import numpy as np
import pandas as pd
from skimage import io
from skimage.feature import hog

warnings.filterwarnings('ignore')


def to_features_and_labels(data):
    train_features = []
    train_labels = []

    for _, image, label in data.to_records():
        img = io.imread('./data/processed/{}.jpeg'.format(image))
        train_features.append(
            hog(img,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                visualise=False))

        train_labels.append(label)

    return train_features, train_labels


def load_dataset(size, test_split=20):
    """
    Load dataset of given size
    :param test_split: Percentage of 'size' data to be used as test.
    :return: train_x, test_x, train_y, test_y
    """
    data = pd.read_csv('./trainLabels.csv')

    total_size = len(data)

    if size > total_size or size < 100:
        raise ValueError('Invalid value for size')

    if not 0 <= test_split <= 50:
        raise ValueError('Invalid percentage for test data')

    test_size = int((test_split / 100) * size)
    train_size = size - test_size

    data_0 = data[data.level == 0]
    data_1 = data[data.level == 1]
    data_2 = data[data.level == 2]
    data_3 = data[data.level == 3]
    data_4 = data[data.level == 4]

    # For TRAIN DATA
    # Calculate size of each rating in sampled dataset
    # We do this while maintaining the same distribution as the full data
    size_0 = int((len(data_0) / total_size) * train_size)
    size_1 = int((len(data_1) / total_size) * train_size)
    size_2 = int((len(data_2) / total_size) * train_size)
    size_3 = int((len(data_3) / total_size) * train_size)
    size_4 = int((len(data_4) / total_size) * train_size)

    # Compensate for rounding errors by adding some images to '0'
    size_0 += (train_size - (size_0 + size_1 + size_2 + size_3 + size_4))

    # Extract training data file names and labels
    train_data_0 = data_0.head(size_0)
    data_0.drop(train_data_0.index.values, inplace=True)

    train_data_1 = data_1.head(size_1)
    data_1.drop(train_data_1.index.values, inplace=True)

    train_data_2 = data_2.head(size_2)
    data_2.drop(train_data_2.index.values, inplace=True)

    train_data_3 = data_3.head(size_3)
    data_3.drop(train_data_3.index.values, inplace=True)

    train_data_4 = data_4.head(size_4)
    data_4.drop(train_data_4.index.values, inplace=True)

    # Read images and extract features
    train_features = []
    train_labels = []

    feats, labels = to_features_and_labels(train_data_0)
    train_features.extend(feats)
    train_labels.extend(labels)

    feats, labels = to_features_and_labels(train_data_1)
    train_features.extend(feats)
    train_labels.extend(labels)

    feats, labels = to_features_and_labels(train_data_2)
    train_features.extend(feats)
    train_labels.extend(labels)

    feats, labels = to_features_and_labels(train_data_3)
    train_features.extend(feats)
    train_labels.extend(labels)

    feats, labels = to_features_and_labels(train_data_4)
    train_features.extend(feats)
    train_labels.extend(labels)

    # For TEST DATA
    # Calculate size of each rating in sampled dataset
    # We do this while maintaining the same distribution as the full data
    size_0 = int((len(data_0) / total_size) * test_size)
    size_1 = int((len(data_1) / total_size) * test_size)
    size_2 = int((len(data_2) / total_size) * test_size)
    size_3 = int((len(data_3) / total_size) * test_size)
    size_4 = int((len(data_4) / total_size) * test_size)

    # Compensate for rounding errors by adding some images to '0'
    size_0 += (test_size - (size_0 + size_1 + size_2 + size_3 + size_4))

    # Extract training data file names and labels
    test_data_0 = data_0.head(size_0)
    test_data_1 = data_1.head(size_1)
    test_data_2 = data_2.head(size_2)
    test_data_3 = data_3.head(size_3)
    test_data_4 = data_4.head(size_4)

    # Read images and extract features
    test_features = []
    test_labels = []

    feats, labels = to_features_and_labels(test_data_0)
    test_features.extend(feats)
    test_labels.extend(labels)

    feats, labels = to_features_and_labels(test_data_1)
    test_features.extend(feats)
    test_labels.extend(labels)

    feats, labels = to_features_and_labels(test_data_2)
    test_features.extend(feats)
    test_labels.extend(labels)

    feats, labels = to_features_and_labels(test_data_3)
    test_features.extend(feats)
    test_labels.extend(labels)

    feats, labels = to_features_and_labels(test_data_4)
    test_features.extend(feats)
    test_labels.extend(labels)

    return (np.array(train_features), np.array(test_features),
            np.array(train_labels), np.array(test_labels))


if __name__ == '__main__':
    X, A, y, b = load_dataset(100)
    print(X.shape, y.shape)
    print(A.shape, b.shape)
