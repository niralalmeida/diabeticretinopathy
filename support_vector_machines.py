"""
 Support Vector Machine Classifier for Diabetic Retinopathy Detection

 Date: 19th February, 2018
"""

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC

from load_dataset import load_dataset


def main():

    train_x, test_x, train_y, test_y = load_dataset(500)

    lda = LinearDiscriminantAnalysis()

    lda.fit(train_x, train_y)
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    lsvc = LinearSVC(random_state=0, dual=False)

    lsvc.fit(train_x, train_y)

    predictions = lsvc.predict(test_x)

    print('Accuracy score: {}'.format(
        metrics.accuracy_score(test_y, predictions)))
    print('Classification Report:\n{}'.format(
        metrics.classification_report(test_y, predictions)))
    print('Confusion Matrix:\n{}'.format(
        metrics.confusion_matrix(test_y, predictions)))


if __name__ == '__main__':
    main()
