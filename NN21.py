#import sys
#sys.path.append('/vols/cms/xw1523/masterproject/prepare_dataframes.py')
#from prepare_dataframes import process_and_save_combined_chunks

print('the very start')

import pandas as pd
print('a')
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def load_data_chunk(chunk_path):
    print(0)
    return pd.read_pickle(chunk_path)

def prepare_data(mc_chunk, combined_chunk):
    # 分配标签
    mc_chunk['label'] = 0
    combined_chunk['label'] = 1
    
    # 合并数据
    data = pd.concat([mc_chunk, combined_chunk], ignore_index=True)
    print(data.head())
    X = data[['Z_mass', 'Z_pt']]
    y = data['label']
    
    return X, y
    print(1)
    
def build_model(input_shape):
    model = Sequential([
        Dense(2,activation='relu', input_dim=2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
    print(2)
    
def update_weights(mc_data, predictions_prob):
    # 计算新重
    predictions_prob=np.minimum(0.9999,predictions_prob)
    new_weights = predictions_prob / (1 - predictions_prob)
    # 更新MC数据的权重
    
    mc_data['new_weight'] = new_weights.flatten() * mc_data['wt']
    return mc_data
    print(3)
    
def plot_weighted_distribution(data, weight_col, feature, bins, label, range=None):
    plt.hist(data[feature], weights=data[weight_col], bins=bins, alpha=0.5, label=label, range=range)
    plt.xlabel(feature)
    plt.ylabel('Weighted Events')
    plt.title(f'Weighted {feature} Distribution')
    plt.legend()

n_chunks = 1
for i in range(n_chunks):
    print('start')
    mc_chunk_path = f'MC_chunks/MC_chunk_{i}.pkl'
    combined_chunk_path = f'combined_chunks/combined_chunk_{i}.pkl'
    
    mc_chunk = load_data_chunk(mc_chunk_path)
    combined_chunk = load_data_chunk(combined_chunk_path)
    
    X, y = prepare_data(mc_chunk, combined_chunk)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
    
    model = build_model(input_shape=(X.shape[1],))
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)  # 设置verbose=0以减少日志输出
    
    print(5)
    
    # 使用模型预测整个MC数据集
    predictions_prob = model.predict(mc_chunk[['Z_mass', 'Z_pt']])
    
    # 更新权重
    mc_chunk_updated = update_weights(mc_chunk, predictions_prob)
    
    print(6)
    
    # 绘制加权的Z_mass和Z_pt分布图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_weighted_distribution(mc_chunk_updated, 'new_weight', 'Z_mass', bins=50, label='MC Weighted', range=[50, 200])

    
    plt.subplot(1, 2, 2)
    plot_weighted_distribution(mc_chunk_updated, 'new_weight', 'Z_pt', bins=50, label='MC Weighted', range=(0, 200))
    plt.legend()
    plt.savefig("NN  Distribution.png")
