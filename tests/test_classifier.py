import unittest

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from irisClassifier import (
    BaseIrisFlowerClassifier,
    NaiveBayesClassifier,
    SVCClassifier,
    LDAClassifier,
)


class ClassifierTestCase(unittest.TestCase):

    def test_create_base_classifier(self):
        classifier = BaseIrisFlowerClassifier()

        assert classifier.cached_species == {
            'setosa': 0,
            'versicolor': 1,
            'virginica': 2,
        }

        assert classifier.reverse_cached_species == {
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        }

        assert classifier.train_observations is not None
        assert classifier.test_observations is not None
        assert classifier.train_responses is not None
        assert classifier.test_responses is not None

        assert classifier.model is None

    def test_naive_bayes(self):
        classifier = NaiveBayesClassifier()

        classifier.train()
        assert isinstance(classifier.model, GaussianNB)
        assert classifier.classify([[6.2, 2.8, 4.8, 1.8]]) == 'virginica'
        assert classifier.classify([[5.5, 2.5, 4.0, 1.3]]) == 'versicolor'
        assert classifier.classify([[5.1, 3.4, 1.5, 0.2]]) == 'setosa'

    def test_svc(self):
        classifier = SVCClassifier()

        classifier.train()
        assert isinstance(classifier.model, SVC)
        assert classifier.classify([[6.2, 2.8, 4.8, 1.8]]) == 'virginica'
        assert classifier.classify([[5.5, 2.5, 4.0, 1.3]]) == 'versicolor'
        assert classifier.classify([[5.1, 3.4, 1.5, 0.2]]) == 'setosa'

    def test_lda(self):
        classifier = LDAClassifier()

        classifier.train()
        assert isinstance(classifier.model, LinearDiscriminantAnalysis)
        assert classifier.classify([[6.2, 2.8, 4.8, 1.8]]) == 'virginica'
        assert classifier.classify([[5.5, 2.5, 4.0, 1.3]]) == 'versicolor'
        assert classifier.classify([[5.1, 3.4, 1.5, 0.2]]) == 'setosa'
