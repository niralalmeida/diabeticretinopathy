"""
 Naive Bayes Classifier for Diabetic Retinopathy Detection

 Date: 18th February, 2018
"""

# from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import metrics

from load_dataset import load_dataset


def main():
    train_x, test_x, train_y, test_y = load_dataset(5000)

    pca = PCA()

    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    gnb = GaussianNB()

    gnb.fit(train_x, train_y)

    predict = gnb.predict(test_x)

    print('Accuracy Score: {}'.format(metrics.accuracy_score(test_y, predict)))
    print('Confusion Matrix:\n{}'.format(
        metrics.confusion_matrix(test_y, predict)))
    print('Report:\n{}'.format(metrics.classification_report(test_y, predict)))


if __name__ == '__main__':
    main()
