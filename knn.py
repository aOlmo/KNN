import random
import numpy as np

from mnist import MNIST

mndata = MNIST('data')


def display_sample(sample):
    print(mndata.display(sample))


class KNN():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def predict(self, k, x):

        distances = []
        argmins = []
        for x_sample in self.X_train:
            distances.append(self.get_distance(x_sample, x))

        for i in range(k):
            argmin_aux = np.argmin(distances)
            distances[argmin_aux] = np.inf
            argmins.append(self.y_train[argmin_aux])

        return np.bincount(argmins).argmax()

    def get_accuracy_of_set(self, K, X_set, y_set):
        n = len(X_set)
        cnt = 0
        for i, elem in enumerate(X_set):
            pred = self.predict(K, elem)
            y = y_set[i]
            cnt += 1 if y == pred else 0
            if i % 200 == 0:
                print("[+]: Iteration {}/{}".format(i, n))

        print("[+]: Accuracy result: {}/{} = {}".format(cnt, n, cnt / n))

if __name__ == '__main__':
    n_train = 6000
    n_test = 1000

    X_train, y_train = mndata.load_training()
    X_test , y_test = mndata.load_testing()

    X_train, y_train = np.array(X_train[:n_train]), np.array(y_train[:n_train])
    X_test, y_test = np.array(X_test[:n_test]), np.array(y_test[:n_test])

    test = KNN(X_train, y_train)
    test.get_accuracy_of_set(99, X_test, y_test)



