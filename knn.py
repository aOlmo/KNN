import random
import numpy as np

import matplotlib.pyplot as plt

from mnist import MNIST

n_runs = 5
n_test = 1000
n_train = 6000
K = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

__RESULTS_DIR__ = "results/"
mndata = MNIST('data')


class KNN():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def predict(self, k, x):

        distances = []
        labels = []
        for x_sample in self.X_train:
            distances.append(self.get_distance(x_sample, x))

        for i in range(k):
            argmin_aux = np.argmin(distances)
            distances[argmin_aux] = np.inf
            labels.append(self.y_train[argmin_aux])

        return np.bincount(labels).argmax()

    def get_accuracy_of_set(self, K, X_set, y_set):
        cnt = 0
        n = len(X_set)
        for i, elem in enumerate(X_set):
            pred = self.predict(K, elem)
            y = y_set[i]
            cnt += 1 if y == pred else 0
            if i % 500 == 0:
                print("[+]: Iteration {}/{}".format(i, n))

        return cnt/n

def display_sample(sample):
    print(mndata.display(sample))

def train_and_get_error_rates():

    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    X_train, y_train = np.array(X_train[:n_train]), np.array(y_train[:n_train])
    X_test, y_test = np.array(X_test[:n_test]), np.array(y_test[:n_test])

    knn = KNN(X_train, y_train)
    train_error_rates = [0] * len(K)
    test_error_rates = [0] * len(K)

    for run in range(n_runs):
        print("==================================")
        print("[+]: RUN {}".format(run + 1))
        print("==================================")

        print("===================")
        print("      TEST SET     ")
        print("===================")
        for i, k in enumerate(K):
            print("------------------")
            print("[+]: Using K = {}".format(k))
            print("------------------")
            accuracy = knn.get_accuracy_of_set(k, X_test, y_test)
            test_error_rates[i] += (1 - accuracy)

        print("===================")
        print("    TRAINING SET   ")
        print("===================")
        for i, k in enumerate(K):
            print("------------------")
            print("[+]: Using K = {}".format(k))
            print("------------------")
            accuracy = knn.get_accuracy_of_set(k, X_train, y_train)
            train_error_rates[i] += (1 - accuracy)

    test_error_rates = np.array(test_error_rates) / float(n_runs)
    train_error_rates = np.array(train_error_rates) / float(n_runs)

    print("TEST")
    print(test_error_rates)

    print("TRAIN")
    print(train_error_rates)

        # np.save(__RESULTS_DIR__ + 'test_error_rates_run_{}'.format(run), test_error_rates)
        # np.save(__RESULTS_DIR__ + 'train_error_rates_run_{}'.format(run), train_error_rates)

if __name__ == '__main__':
    # train_and_get_error_rates()

    test_error_rates = np.load(__RESULTS_DIR__+"test_error_rates_run_1.npy")
    train_error_rates = np.load(__RESULTS_DIR__+"train_error_rates_run_1.npy")

    plt.title("Error rates")
    plt.plot(K, test_error_rates, 'bo-', label='Test set', linewidth=2)
    plt.plot(K, train_error_rates, 'ro-', label='Train set', linewidth=2)
    plt.legend(loc='lower right')
    plt.show()

