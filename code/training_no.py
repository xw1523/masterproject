import json
import os
import pickle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import numpy as np
import time

# Load and process data
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
    # Concatenate all data
    return np.concatenate(features_all), np.concatenate(labels_all), np.concatenate(weights_all)

# List all the training and validation data paths
train_paths = [
    '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_0.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/train_chunks/train_chunk_1.pkl'
]
val_paths = [
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_0.pkl',
    '/vols/cms/yl13923/masterproject/new_data_chunks/test_chunks/test_chunk_1.pkl'
]

# Load training and validation data
X_train, y_train, weights_train = load_and_prepare_data(train_paths)
X_val, y_val, weights_val = load_and_prepare_data(val_paths)

# Memory monitoring callback
class MemoryProfilerCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.save_dir = '/vols/cms/yl13923/masterproject/memory_plots_no_2'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
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
        if current_time - self.last_recorded_time >= 2:
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
        if current_time - self.last_saved_time >= 30:
            filename = os.path.join(self.save_dir, f'memory_usage_plot_{self.update_count}.png')
            self.fig.savefig(filename)
            self.last_saved_time = current_time
            self.update_count += 1

    def on_train_end(self, logs=None):
        plt.ioff()
        final_filename = os.path.join(self.save_dir, 'final_memory_usage_plot.png')
        self.fig.savefig(final_filename)
        plt.close(self.fig)

# Define the model
def simple_model(input_dimension):
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.summary()
    return model

# Initialize the model
model = simple_model(X_train.shape[1])

# Train the model
history = model.fit(
    X_train, 
    y_train, 
    sample_weight=weights_train, 
    batch_size=6000,
    epochs=10, 
    validation_data=(X_val, y_val, weights_val), 
    callbacks=[MemoryProfilerCallback()]
)

# Save the model
model_save_path = '/vols/cms/yl13923/masterproject/my_model_Zphi.h5'
model.save(model_save_path)

# Save training history
history_save_path = '/vols/cms/yl13923/masterproject/train_history_Zphi.json'
with open(history_save_path, 'w') as f:
    json.dump(history.history, f)
