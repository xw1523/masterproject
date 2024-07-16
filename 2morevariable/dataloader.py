import pandas as pd
import numpy as np
import tensorflow as tf

#get generator
features = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj', 'jdeta', 'jdphi', 'dijetpt', 'jpt_1', 'jpt_2', 'jpt_3']


def get_generator(file_paths):
    def _generator():
        for file_path in file_paths:
            data = pd.read_pickle(file_path)
            X = data[features].astype(np.float32)
            y = data['label'].astype(np.int32)
            weights = data.get('wt', np.ones(len(data))).astype(np.float32)

            for i in range(len(data)):
                yield (X.iloc[i], y.iloc[i], weights[i])

            del data, X, y, weights

    return _generator