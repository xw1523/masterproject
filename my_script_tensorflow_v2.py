import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History

shuffled_chunk_paths = [
    '/vols/cms/yl13923/masterproject/shuffled_chunk1.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk2.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk3.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk4.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk5.pkl'
]


def simple_model(input_dimension):
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(input_dimension*2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    optimizer =  Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  
    return model


# def model
def train_model_on_chunk(chunk_path, version):
    chunk = pd.read_pickle(chunk_path)
    X = chunk[['Z_mass', 'Z_pt']].values
    y = chunk['label'].values
    weights = chunk['wt'].values 

    print(1)

    scaler_x = MinMaxScaler()
    X = scaler_x.fit_transform(X)
    
    print(2)

    # define early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=4)


    # split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, 
        y, 
        weights,
        test_size=0.90, 
        random_state=42,
        stratify=y)

    print(3)

    history = History()
    model = simple_model(X_train.shape[1])
    
    print(4)

    # train
    model.fit(X_train, 
              y_train, 
              sample_weight=weights_train,
              batch_size=32,
              epochs=15, 
              callbacks=[history,early_stop],
              validation_data=(X_test, y_test)) 

    model_save_path = chunk_path.replace('.pkl', f'_v{version}_model.h5')
    model.save(model_save_path)

    history_path = chunk_path.replace('.pkl', f'_v{version}_history.json')
    with open(history_path, 'w') as file:
        json.dump(history.history, file)

    y_pred = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)

    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc_score': auc_score
    }
    roc_data_path = chunk_path.replace('.pkl', f'_v{version}_roc_data.json')
    with open(roc_data_path, 'w') as file:
        json.dump(roc_data, file)

    print(f"Training completed for {chunk_path}. Model, history, and ROC data saved.")

# operate on each data chunk
version_number = '2' 
for chunk_path in shuffled_chunk_paths:
    train_model_on_chunk(chunk_path, version_number)