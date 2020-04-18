from utils.model import split_method, cv_method
import numpy as np
from utils.data import load_data, preprocessing_data
from utils.plot import plot_optimal_k,plot_data
from utils.model import find_optimal_k, training_methods, latest_modified_weight, load_model, train_model
import argparse


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  '
                             'False: Train model default: False ')

    parser.add_argument('--method', choices=list(training_methods.keys()),
                        help='Training methods: cv-  Cross-Validation  default 10-Fold '
                             'split- default 70 perc train, 30 perc test')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(3)

    args = parse_args()

    if args.load:
        weight = latest_modified_weight()
        model = load_model(weight)

        income, age, loan = 27845.8008938469, 55.4968525394797, 10871.1867897838
        print(model.predict([list((income, age, loan))])[0])
    else:

        features, labels = load_data('data/credit_data.csv', 'default', ('income', 'age', 'loan'))

        plot_data(features,labels)

        features = preprocessing_data(features)

        k_range, scores, opt_k, opt_score = find_optimal_k(features, labels)
        plot_optimal_k(k_range, scores, opt_k, opt_score)

        model = train_model(features, labels, opt_k, args.method)

        income, age, loan = 27845.8008938469, 55.4968525394797, 10871.1867897838
        print(model.predict([list((income, age, loan))])[0])
