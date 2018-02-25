"""
 Random Forests Classifier for Diabetic Retinopathy Detection

 Date: 19th February, 2018
"""

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from load_dataset import load_dataset


def main():
    train_x, test_x, train_y, test_y = load_dataset(5000)

    rfc = RandomForestClassifier(random_state=0)

    rfc.fit(train_x, train_y)

    predictions = rfc.predict(test_x)

    print('Accuracy score: {}'.format(
        metrics.accuracy_score(test_y, predictions)))
    print('Classification Report:\n{}'.format(
        metrics.classification_report(test_y, predictions)))
    print('Confusion Matrix:\n{}'.format(
        metrics.confusion_matrix(test_y, predictions)))


if __name__ == '__main__':
    main()
