import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(path, label, *features):
    df = pd.read_csv(path)
    features, labels = df[list(*features)], df[label]
    return features, labels


def preprocessing_data(features):
    return MinMaxScaler().fit_transform(features)
