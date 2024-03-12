import uproot
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import History 


final_chunk=['/vols/cms/xw1523/masterproject/final_chunks/final_chunk_0.pkl',
             '/vols/cms/xw1523/masterproject/final_chunks/final_chunk_1.pkl',
             '/vols/cms/xw1523/masterproject/final_chunks/final_chunk_2.pkl',
             '/vols/cms/xw1523/masterproject/final_chunks/final_chunk_3.pkl',
             '/vols/cms/xw1523/masterproject/final_chunks/final_chunk_4.pkl']


# define a simple NN
def baseline_model(input_dimension):
    # create model
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(input_dimension*2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model



def prepare_data(chunk_path):
    merged_dff=pd.read_pickle(chunk_path)
    print("Loaded DataFrame columns:", merged_dff.columns)  # 打印列名
    X = merged_dff[['Z_mass', 'Z_pt']]
    y = merged_dff['label']
    weights = merged_dff['wt']
    

    
    scaler_x = MinMaxScaler()
    X=scaler_x.fit_transform(X)
    
    # define early stopping
    early_stop = EarlyStopping(monitor='val_loss',patience=10)
    
    X_train,X_test, y_train, y_test, weights_train, weights_test   = train_test_split(
    X,
    y,
    weights,
    test_size=0.95,
    random_state=46,
    stratify=y.values,
)
    
    history = History()

    model = baseline_model(X_train.shape[1])


    #model.fit(
               # X_train, y_train,
                #sample_weight=weights_train,
               # batch_size=32,
              #  epochs=10,
               # callbacks=[history,early_stop],
                #validation_data=(X_test, y_test))
    
    # 将训练和测试数据转换为tf.data.Dataset对象
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, weights_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test, weights_test)).batch(32)
    
    # 使用包含权重的Dataset进行训练和验证
    model.fit(
        train_dataset,
        epochs=10,
        callbacks=[history, early_stop],
        validation_data=test_dataset
    )
    
    model_save_path=chunk_path.replace('.pkl', '_model11.h5')
    model.save(model_save_path)   
    
    
     # 保存训练历史
    history_path = chunk_path.replace('.pkl', '_history.json')
    with open(history_path, 'w') as file:
        json.dump(history.history, file)


    # 使用模型进行预测
    y_pred_probs = model.predict(X_test).ravel()
    
    # 计算ROC曲线参数和AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    auc_score = roc_auc_score(y_test, y_pred_probs)
    
    # 保存ROC数据
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc_score': auc_score
    }
    roc_data_path = chunk_path.replace('.pkl', '_roc_data.json')
    with open(roc_data_path, 'w') as file:
        json.dump(roc_data, file)
  
    
    print(f"Traning completed for {chunk_path}. Model,history,roc saved. ") 
    
    
for chunk_path in final_chunk:
   prepare_data(chunk_path) 