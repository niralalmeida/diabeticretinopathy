"""
 Naive Bayes Classifier for Diabetic Retinopathy Detection

 Date: 18th February, 2018
"""

from load_sample_dataset import load_sample_dataset

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def main():
    train_x, test_x, train_y, test_y = load_sample_dataset()

    gnb = GaussianNB()

    gnb.fit(train_x, train_y)

    predictions = gnb.predict(test_x)

    print('Accuracy score: {}'.format(
        metrics.accuracy_score(test_y, predictions)))
    print('Classification Report:\n{}'.format(
        metrics.classification_report(test_y, predictions)))
    print('Confusion Matrix:\n{}'.format(
        metrics.confusion_matrix(test_y, predictions)))


if __name__ == '__main__':
    main()
