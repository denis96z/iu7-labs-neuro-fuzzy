import sys

import numpy as np
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier


def main(args):
    assert len(args) == 2

    tr_path = args[0]
    ts_path = args[1]

    tr_set = np.loadtxt(fname=tr_path, delimiter=",")
    x_tr = tr_set[:, 1:]
    y_tr = tr_set[:, 0]

    x_tr_n = preprocessing.normalize(x_tr)

    model = KNeighborsClassifier()
    model.fit(x_tr_n, y_tr)

    ts_set = np.loadtxt(fname=ts_path, delimiter=",")
    x_ts = ts_set[:, 1:]
    y_ts = ts_set[:, 0]

    x_ts_n = preprocessing.normalize(x_ts)

    expected = y_ts
    predicted = model.predict(x_ts_n)

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


if __name__ == "__main__":
    main(sys.argv[1:])
