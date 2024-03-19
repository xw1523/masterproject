import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History
import tensorflow as tf

shuffled_chunk_paths = [
    '/vols/cms/yl13923/masterproject/shuffled_chunk1.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk2.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk3.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk4.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk5.pkl'
]

# Split data into training sets and validation sets
def split_and_save_chunks(chunk_paths, output_dir, test_size=0.9):
    for i, chunk_path in enumerate(chunk_paths):
        data_chunk = pd.read_pickle(chunk_path)
        
        train_chunk, val_chunk = train_test_split(data_chunk, test_size=test_size, random_state=42)
        
        train_chunk_path = os.path.join(output_dir, f"train_chunk_{i+1}.pkl")
        val_chunk_path = os.path.join(output_dir, f"val_chunk_{i+1}.pkl")
        
        train_chunk.to_pickle(train_chunk_path)
        val_chunk.to_pickle(val_chunk_path)

output_dir = '/vols/cms/yl13923/masterproject/splitted_chunks'

split_and_save_chunks(shuffled_chunk_paths, output_dir)

train_paths = [
     '/vols/cms/yl13923/masterproject/splitted_chunks/train_chunk_1.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/train_chunk_2.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/train_chunk_3.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/train_chunk_4.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/train_chunk_5.pkl'
]
               
val_paths = [
     '/vols/cms/yl13923/masterproject/splitted_chunks/val_chunk_1.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/val_chunk_2.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/val_chunk_3.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/val_chunk_4.pkl',
     '/vols/cms/yl13923/masterproject/splitted_chunks/val_chunk_5.pkl'
]

def get_generator(file_paths):
    scaler = MinMaxScaler() 
    def _generator():
        for file_path in file_paths:
            df = pd.read_pickle(file_path)
            X = df[['Z_mass', 'Z_pt']].values
            y = df['label'].values
            weights = df['wt'].values

            X_scaled = scaler.fit_transform(X)

            for i in range(len(df)):
                yield (X_scaled[i], y[i], weights[i])
    return _generator

train_generator = get_generator(train_paths)
val_generator = get_generator(val_paths)

print(1)

data_train = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(32)

print(2)

data_val = tf.data.Dataset.from_generator(
    val_generator,
    output_signature=(
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(32)

print(3)

def simple_model(input_dimension):
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(input_dimension*2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    optimizer =  Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  
    return model


model = simple_model(2)

print(4)

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    data_train, 
    validation_data=data_val,
    epochs=10, 
    callbacks=[early_stop]
)
print(5)

# save model
model_save_path = '/vols/cms/yl13923/masterproject/my_model.h5'
model.save(model_save_path)

print(6)

history_save_path = '/vols/cms/yl13923/masterproject/train_history.json'
with open(history_save_path, 'w') as f:
    json.dump(history.history, f)

print(7)

# ROC
y_test_combined = []
y_pred_combined = []
for val_path in val_paths:
    val_data = pd.read_pickle(val_path)
    X_val = val_data[['Z_mass', 'Z_pt']].values
    y_val = val_data['label'].values
    X_val_scaled = scaler.transform(X_val) 
    y_pred = model.predict(X_val_scaled).ravel()
    
    y_test_combined.extend(y_val)
    y_pred_combined.extend(y_pred)

print(8)

test_labels_and_predictions = {'y_test': y_test_combined, 'y_pred': y_pred_combined}
test_labels_and_predictions_path = '/vols/cms/yl13923/masterproject/val_labels_and_predictions.pkl'
pd.to_pickle(test_labels_and_predictions, test_labels_and_predictions_path)