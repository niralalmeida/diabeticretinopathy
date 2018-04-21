"""
 Naive Bayes Classifier for Diabetic Retinopathy Detection

 Results after 5-fold Cross Validation
 [0.79268293 0.80246914 0.8125 0.88607595 0.84615385]

 Date: 18th February, 2018
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from load_dataset import load_dataset


def main():
    train_x, test_x, train_y, test_y = load_dataset(5000)

    lda = LinearDiscriminantAnalysis()

    lda.fit(train_x, train_y)
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    scores = cross_val_score(
        GaussianNB(), train_x, train_y, scoring='accuracy', cv=5)

    print(scores)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
