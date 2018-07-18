"""
 Random Forests Classifier for Diabetic Retinopathy Detection

 Hyperparameter search using Grid Search

 Date: 19th February, 2018
"""

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

from load_dataset import load_dataset


def main():

    train_x, test_x, train_y, test_y = load_dataset(500)

    lda = LinearDiscriminantAnalysis()

    lda.fit(train_x, train_y)
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    parameters = {
        "n_estimators": [10, 50, 100, 500],
        "criterion": ["gini", "entropy"],
        "max_features": ["log2", "sqrt", "auto"],
        "max_depth": [None, 10, 25, 50]
    }

    gd = GridSearchCV(RandomForestClassifier(), parameters, scoring="accuracy")

    gd.fit(train_x, train_y)

    print('Best Score: {}'.format(gd.best_score_))

    print('Best Model:\n{}'.format(gd.best_estimator_))

    rfc = gd.best_estimator_

    pred = rfc.predict(test_x)

    print('On Test Data...')

    print('Accuracy: {}'.format(metrics.accuracy_score(test_y, pred)))


if __name__ == '__main__':
    main()
