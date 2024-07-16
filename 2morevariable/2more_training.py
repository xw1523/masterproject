import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
import json
import tensorflow as tf
from dataloader import get_generator  



test_chunk_paths = [
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk0.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk1.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk2.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk3.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk4.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk5.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk6.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk7.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk8.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk9.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk10.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk11.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk12.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk13.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk14.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk15.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk16.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk17.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk18.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk19.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk20.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk21.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk22.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk23.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/test_scaled_chunk24.pkl'
]

train_chunk_paths = [
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk0.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk1.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk2.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk3.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk4.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk5.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk6.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk7.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk8.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk9.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk10.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk11.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk12.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk13.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk14.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk15.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk16.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk17.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk18.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk19.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk20.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk21.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk22.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk23.pkl',
    '/vols/cms/xw1523/masterproject/2morevariable/pickle_files/train_scaled_chunk24.pkl'
]



# Define features for the model
features = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj', 'jdeta', 'jdphi', 'dijetpt', 'jpt_1', 'jpt_2', 'jpt_3']


batch_size=1000


train_generator = get_generator(train_chunk_paths)
test_generator = get_generator(test_chunk_paths)


#n_batches_train = n_events_train//batch_size
# TEMPORARY: test numbers < 60000, and see when problem appears
# TODO: Disable early stopping and callbacks -> make model simple

n_batches_train = 9000
n_batches_val = 8000

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        tf.TensorSpec(shape=(len(features),), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(1000).prefetch(tf.data.AUTOTUNE).take(n_batches_train)

test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=(
        tf.TensorSpec(shape=(len(features),), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(1000).prefetch(tf.data.AUTOTUNE).take(n_batches_val)


# # Test to see if generators behaving ok:
# i_train = iter(train_dataset)
# for b in range(n_batches_train):
#     next(i_train)
#     print(f"Batch {b} is ok!")

# raise RuntimeError("stop")

#training 
# Define the model architecture with input dimension parameter
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

def baseline_model(input_dimension):
    model = Sequential([
        Input(shape=(input_dimension,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(1, activation="sigmoid")
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                  metrics=['accuracy'])
    model.summary()
    return model


# 创建模型时传递特征数量
model = baseline_model(len(features))
history = History()
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)


# Train the model
model.fit(
    train_dataset,
    epochs=80,
    # callbacks=[history, early_stop, reduce_lr], # disable for now!
    steps_per_epoch = n_batches_train,
    validation_data=test_dataset,
    validation_steps=n_batches_val
)


chunk_path = "/vols/cms/xw1523/masterproject/2morevariable/final_modeltrainingfull"

# 保存模型
model_save_path = '/vols/cms/xw1523/masterproject/2morevariable/final_modelgentrainingfull.h5'
model.save(model_save_path)
# 保存模型
model_save_path = '/vols/cms/xw1523/masterproject/2morevariable/final_modelgentrainingfull.keras'
model.save(model_save_path)

# 保存训练历史
history_path = '/vols/cms/xw1523/masterproject/2morevariable/historygentrainingfull.json'
with open(history_path, 'w') as file:
    json.dump(history.history, file)
 
