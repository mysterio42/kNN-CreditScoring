import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from utils.plot import plot_cm
import string
from sklearn.model_selection import cross_validate
import glob
from operator import itemgetter
import os
import joblib

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model(path):
    """

    :param path: weight path
    :return: load model based on the path
    """
    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def find_optimal_k(features, labels):
    """

    :param features: 'income'', 'age', 'loan'
    :param labels: 'default'
    :return:
    """
    scores = []
    k_range = range(1, 101)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, features, labels, cv=10, scoring='accuracy')
        scores.append(score.mean())
    opt_k = np.argmax(scores) + 1  # because index starts from zero !
    opt_score = np.amax(scores)
    return k_range, scores, opt_k, opt_score


def split_method(features, labels, opt_k):
    """
    run train_test_split method with optimal K
    :param features: 'income'', 'age', 'loan'
    :param labels: 'default'
    :return:
    """
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=opt_k)
    model.fit(feature_train, label_train)

    predictions = model.predict(feature_test)

    cm = confusion_matrix(label_test, predictions)

    plot_cm(cm,'split')

    print(accuracy_score(label_test, predictions))

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        model_name = WEIGHTS_DIR + 'knn-' + generate_model_name(5) + '.pkl'
        with open(model_name, 'wb') as f:
            joblib.dump(value=model, filename=f, compress=3)
            print(f'Model saved at {model_name}')

    return model


def cv_method(features, labels, opt_k):
    """
    run Cross-Validation method with optimal K
    :param features: 'income'', 'age', 'loan'
    :param labels: 'default'
    :return:
    """
    cv_results = cross_validate(KNeighborsClassifier(n_neighbors=opt_k),
                                features, labels,
                                return_estimator=True, return_train_score=True, cv=10)
    estimator_testscore = zip(list(cv_results['estimator']), cv_results['train_score'])
    model = max(estimator_testscore, key=itemgetter(1))[0]

    preds = model.predict(features)
    cm = confusion_matrix(labels, preds)
    plot_cm(cm,'cv')

    print(accuracy_score(labels, preds))

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        model_name = WEIGHTS_DIR + 'knn-' + generate_model_name(5) + '.pkl'
        with open(model_name, 'wb') as f:
            joblib.dump(value=model, filename=f, compress=3)
            print(f'Model saved at {model_name}')

    return model


def train_model(features, labels, opt_k, method):
    """
    run appropriate training method
    :param features: 'income'', 'age', 'loan'
    :param labels: 'default'
    :param method: 'cv' for Cross-Validation method , 'split' for train_test_split method
    :return:
    """
    return training_methods[method](features, labels, opt_k)


training_methods = {
    'split': split_method,
    'cv': cv_method
}
