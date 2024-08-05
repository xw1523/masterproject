import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import time
from keras import Sequential, backend as K
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from memory_profiler import memory_usage
from sklearn.metrics import roc_curve, roc_auc_score
from kerastuner import HyperModel, RandomSearch
from keras.initializers import RandomNormal

version = "2"

# Function to create directories
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to create versioned paths for file saving
def get_versioned_path(base_path, subfolder, file_name, version):
    path = os.path.join(base_path, subfolder)
    create_directory(path)
    versioned_file_name = f"{os.path.splitext(file_name)[0]}_version_{version}{os.path.splitext(file_name)[1]}"
    return os.path.join(path, versioned_file_name)

# Function to load data
def load_and_prepare_data(file_paths):
    features_all, labels_all, weights_all = [], [], []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        features = data.drop(columns=['met', 'met_phi', 'label', 'wt'])
        labels = data['label']
        weights = data['wt']
        features_all.append(features.values)
        labels_all.append(labels.values)
        weights_all.append(weights.values)
    
        del data, features, labels, weights
    combined_features = np.concatenate(features_all)
    combined_labels = np.concatenate(labels_all)
    combined_weights = np.concatenate(weights_all)
    
    del features_all, labels_all, weights_all
    return combined_features, combined_labels, combined_weights

# Memory monitoring callback
class MemoryProfilerCallback(Callback):
    def on_train_begin(self, logs=None):
        self.save_dir = '/vols/cms/yl13923/masterproject/memory_plots_no'
        create_directory(self.save_dir)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.mem_usage = []
        self.times = []
        self.start_time = time.time()
        self.last_recorded_time = self.start_time
        self.last_saved_time = self.start_time
        self.update_count = 0

    def on_batch_end(self, batch, logs=None):
        current_time = time.time()
        if current_time - self.last_recorded_time >= 60:  # Record per 1 min
            mem_usage = memory_usage(-1, interval=0.1, timeout=1)[0]
            elapsed_time = (current_time - self.start_time) / 60
            self.mem_usage.append(mem_usage)
            self.times.append(elapsed_time)
            self.last_recorded_time = current_time
            self.ax.clear()
            self.ax.plot(self.times, self.mem_usage)
            self.ax.set_title('Memory Usage over Time')
            self.ax.set_xlabel('Time (minutes)')
            self.ax.set_ylabel('Memory Usage (MiB)')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        if current_time - self.last_saved_time >= 600:  # Save per 10 min
            filename = os.path.join(self.save_dir, f'memory_usage_plot_{self.update_count}.png')
            self.fig.savefig(filename)
            self.last_saved_time = current_time
            self.update_count += 1

    def on_train_end(self, logs=None):
        plt.ioff()
        final_filename = os.path.join(self.save_dir, 'final_memory_usage_plot.png')
        self.fig.savefig(final_filename)
        plt.close(self.fig)

class MyHyperModel(HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32),
                        activation='relu', input_dim=self.input_dim, kernel_initializer=RandomNormal()))
        model.add(Dropout(hp.Float('dropout1', min_value=0, max_value=0.5, step=0.1)))
        model.add(BatchNormalization())
        
        model.add(Dense(units=hp.Int('units2', min_value=32, max_value=256, step=32), activation='relu',
                        kernel_initializer=RandomNormal()))
        model.add(Dropout(hp.Float('dropout2', min_value=0, max_value=0.5, step=0.1)))
        model.add(BatchNormalization())

        model.add(Dense(units=hp.Int('units3', min_value=32, max_value=128, step=32), activation='relu',
                        kernel_initializer=RandomNormal()))
        model.add(Dropout(hp.Float('dropout3', min_value=0, max_value=0.5, step=0.1)))
        model.add(BatchNormalization())

        model.add(Dense(units=hp.Int('units4', min_value=16, max_value=128, step=16), activation='relu',
                        kernel_initializer=RandomNormal()))
        model.add(Dropout(hp.Float('dropout4', min_value=0, max_value=0.5, step=0.1)))
        model.add(BatchNormalization())

        model.add(Dense(1, activation='sigmoid', kernel_initializer=RandomNormal()))
        
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

# Load training and validation data
train_paths = [
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_0.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_1.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_2.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_3.pkl', 
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_4.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_5.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_6.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_7.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_8.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_9.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_10.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_11.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_12.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_13.pkl', 
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_14.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_15.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_16.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_17.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_18.pkl',
        '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_19.pkl'
]
val_paths = [
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_0.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_1.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_2.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_3.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_4.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_5.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_6.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_7.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_8.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_9.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_10.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_11.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_12.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_13.pkl', 
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_14.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_15.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_16.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_17.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_18.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_19.pkl'
]
X_train, y_train, weights_train = load_and_prepare_data(train_paths)
X_val, y_val, weights_val = load_and_prepare_data(val_paths)

# Initialize the HyperModel
hypermodel = MyHyperModel(input_dim=X_train.shape[1])

# 初始化早停机制的设置
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=3,  
    verbose=1, 
    restore_best_weights=True  # 恢复最佳模型的权重
)

# Configure the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory=os.path.join('/vols/cms/yl13923/masterproject', 'tuner'),
    project_name='hypermodel_tuning'
)

# Start hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[MemoryProfilerCallback(), early_stopping])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0].values
print("Best hyperparameters:\n", best_hyperparameters)

# Save the best hyperparameters to a JSON file
hyperparams_path = get_versioned_path('/vols/cms/yl13923/masterproject/hyperparams', '', 'best_hyperparameters.json', version)
with open(hyperparams_path, 'w') as f:
    json.dump(best_hyperparameters, f)

# Plot ROC curve for the best model
y_pred = best_model.predict(X_val).ravel()
fpr, tpr, _ = roc_curve(y_val, y_pred, sample_weight=weights_val)
auc_score = roc_auc_score(y_val, y_pred, sample_weight=weights_val)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
roc_curve_path = get_versioned_path('/vols/cms/yl13923/masterproject/ROC', '', 'roc_curve.png', version)
plt.savefig(roc_curve_path)
plt.close()

# Save the model and history
model_save_path = get_versioned_path('/vols/cms/yl13923/masterproject/models', '', 'best_model.h5', version)
best_model.save(model_save_path)

history_save_path = get_versioned_path('/vols/cms/yl13923/masterproject/train_history', '', 'train_history.json', version)
with open(history_save_path, 'w') as f:
    json.dump(best_model.history.history, f)
