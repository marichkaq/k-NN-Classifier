import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    data = datasets.load_iris()
    X = data.data  # attributes
    y = data.target  # labels

    # split dataset into a training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k_range = range(1, 101)
    accuracies = []

    for k in k_range:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)                # train classifier

        score = classifier.score(X_test, y_test)        # evaluate an add accuracy
        accuracies.append(score)

    plt.plot(k_range, accuracies, linestyle='-',  marker=None)   # plot the results
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.title('k-NN: Varying Number of Neighbors')
    plt.show()

if __name__ == "__main__":
    main()

