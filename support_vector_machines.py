"""
 Support Vector Machine Classifier for Diabetic Retinopathy Detection

 Date: 19th February, 2018
"""

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from load_dataset import load_dataset


def main():

    train_x, test_x, train_y, test_y = load_dataset(500)

    lda = LinearDiscriminantAnalysis()

    lda.fit(train_x, train_y)
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    parameters = [
        {
            'C': [1.0, 2.0, 5.0],
            'dual': [True],
            'penalty': ['l2'],
            'loss': ['hinge', 'squared_hinge']
        },
        {
            'C': [1.0, 2.0, 5.0],
            'dual': [False],
            'penalty': ['l1'],
            'loss': ['squared_hinge']
        }
    ]

    gd = GridSearchCV(LinearSVC(), parameters, scoring="accuracy")

    gd.fit(train_x, train_y)

    print('Best Score: {}'.format(gd.best_score_))

    print('Best Model:\n{}'.format(gd.best_estimator_))

    lsvc = gd.best_estimator_

    pred = lsvc.predict(test_x)

    print('On Test Data...')

    print('Accuracy: {}'.format(metrics.accuracy_score(test_y, pred)))


if __name__ == '__main__':
    main()
