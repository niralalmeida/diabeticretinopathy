"""
 Random Forests Classifier for Diabetic Retinopathy Detection

 Date: 19th February, 2018
"""


from sklearn.svm import LinearSVC
from sklearn import metrics

from load_sample_dataset import load_sample_dataset


def main():
    train_x, test_x, train_y, test_y = load_sample_dataset()

    lsvc = LinearSVC(random_state=0)

    lsvc.fit(train_x, train_y)

    predictions = lsvc.predict(test_x)

    print('Accuracy score: {}'.format(metrics.accuracy_score(test_y, predictions)))
    print('Classification Report:\n{}'.format(metrics.classification_report(test_y, predictions)))
    print('Confusion Matrix:\n{}'.format(metrics.confusion_matrix(test_y, predictions)))


if __name__ == '__main__':
    main()
