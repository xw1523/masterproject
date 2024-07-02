import pandas as pd

def get_generator(file_paths):
    def generator():
        for file_path in file_paths:
            df = pd.read_pickle(file_path)
            X = df[['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj', 'jdeta', 'jdphi', 'dijetpt', 'jpt_1', 'jpt_2', 'jpt_3','U1','U2','met','met_phi']].values
            y = df['label'].values
            weights = df['wt'].values

            for i in range(len(df)):
                yield (X[i], y[i], weights[i])
    return generator