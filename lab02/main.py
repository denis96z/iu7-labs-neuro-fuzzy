import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

NUM_CLUSTERS = 10

TR_DS_LIMIT = 200
TS_DS_LIMIT = 100


def main(args):
    if len(args) != 2:
        raise RuntimeError('not enough arguments')

    tr_path = args[0]
    ts_path = args[1]

    tr_set = np.loadtxt(fname=tr_path, delimiter=',')
    if TR_DS_LIMIT is None:
        x_tr = tr_set[:, 1:]
        y_tr = tr_set[:, 0]
    else:
        x_tr = tr_set[:TR_DS_LIMIT, 1:]
        y_tr = tr_set[:TR_DS_LIMIT, 0]

    ts_set = np.loadtxt(fname=ts_path, delimiter=',')
    if TS_DS_LIMIT is None:
        x_ts = ts_set[:, 1:]
        y_ts = ts_set[:, 0]
    else:
        x_ts = ts_set[:TS_DS_LIMIT, 1:]
        y_ts = ts_set[:TS_DS_LIMIT, 0]

    sc = StandardScaler()
    x_tr = sc.fit_transform(x_tr)
    x_ts = sc.fit_transform(x_ts)

    data = {'precision_score': [], 'recall_score': [], 'f1_score': []}

    for i in range(2, NUM_CLUSTERS + 1):
        k_means = KMeans(n_clusters=i, random_state=0)
        k_means.fit(x_tr, y_tr)
        labels__ = k_means.predict(x_ts)
        data['precision_score'].append(precision_score(y_ts, labels__, average='weighted'))
        data['recall_score'].append(recall_score(y_ts, labels__, average='micro'))
        data['f1_score'].append(f1_score(y_ts, labels__, average='macro'))

    results = pd.DataFrame(data=data, index=range(2, NUM_CLUSTERS + 1))
    print(results)


if __name__ == '__main__':
    main(sys.argv[1:])
