import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class BaseIrisFlowerClassifier(object):

    def __init__(self):
        self.irises = pd.read_csv(
            os.path.join(BASE_PATH, 'data/iris.csv')
        )

        feature_names = [
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ]

        self.observations = self.irises[feature_names]

        self.cached_species = {
            pair[0]: pair[1]
            for pair in zip(self.irises['species'].unique(), xrange(3))
        }

        self.reverse_cached_species = {
            value: key for key, value in self.cached_species.items()
        }

        self.irises['species_id'] = [
            self.cached_species[species]
            for species in self.irises['species']
        ]

        self.responses = self.irises['species_id']

        (
            self.train_observations,
            self.test_observations,
            self.train_responses,
            self.test_responses,
        ) = train_test_split(self.observations, self.responses)

        self.model = None

    def train(self):
        """ Train classifier. Implemented in child classes."""
        raise NotImplementedError

    def score(self):
        """ Print scoring of selected method. """
        print('Accuracy of classifier on training set: {:.2f}'.format(
            self.model.score(self.train_observations, self.train_responses))
        )
        print('Accuracy of SVM classifier on test set: {:.2f}'.format(
            self.model.score(self.train_observations, self.train_responses))
        )

    def classify(self, observations):
        """Classify given observations.

        :param observations list [
                'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
            ]
        :return string, name of species
        """
        result = self.model.predict(observations)
        return self.reverse_cached_species[result[0]]


class NaiveBayesClassifier(BaseIrisFlowerClassifier):

    def __init__(self):
        super(NaiveBayesClassifier, self).__init__()

    def train(self):
        from sklearn.naive_bayes import GaussianNB
        self.model = GaussianNB()
        self.model.fit(self.train_observations, self.train_responses)


class SVCClassifier(BaseIrisFlowerClassifier):

    def train(self):
        from sklearn.svm import SVC
        self.model = SVC()
        self.model.fit(self.train_observations, self.train_responses)


class LDAClassifier(BaseIrisFlowerClassifier):

    def train(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.model = LinearDiscriminantAnalysis()
        self.model.fit(self.train_observations, self.train_responses)


def main(classifier_name, number, show_accuracy):
    classifier = {
        'naive_bayes': NaiveBayesClassifier(),
        'svc': SVCClassifier(),
        'lda': LDAClassifier(),
    }.get(classifier_name)

    classifier.train()

    if show_accuracy:
        classifier.score()

    for i in xrange(number):

        while True:
            # Wait here as long as values are not correct
            print 'Insert values separated by space[sepal_length sepal_width petal_length petal_width]'
            line = raw_input()
            try:
                input = [float(x) for x in line.split()]
            except ValueError:
                print "Only numbers are allowed"
                continue

            if len(input) != 4:
                print "You need to insert 4 numbers"
                continue

            # continue
            break

        print classifier.classify([input])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iris flower classifier')
    parser.add_argument(
        'classifier',
        choices=[
            'naive_bayes',
            'svc',
            'lda',
        ])
    parser.add_argument('number', type=int, default=1)
    parser.add_argument('--show-accuracy', action="store_true")

    args = parser.parse_args()
    main(
        classifier_name=args.classifier,
        number=args.number,
        show_accuracy=args.show_accuracy
    )
