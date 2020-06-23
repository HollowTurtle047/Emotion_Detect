
import csv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import numpy as np
import os

with open('fer2013.csv') as f:
    f_csv = csv.DictReader(f)
    
    # Initialize
    train_images = []
    train_labes = [] 
    test_images = []
    test_labes = [] 
    
    for row in f_csv:
        # emotion,pixels,Usage
        emotion = row["emotion"]
        pixels = row["pixels"]
        usage = row["Usage"]
        
        temp_x = list(map(int, pixels.split()))
        temp_y = int(emotion)
        
        if usage == "Training":
            train_images.append(temp_x)
            train_labes.append(temp_y)
            
        elif usage == "PublicTest":
            test_images.append(temp_x)
            test_labes.append(temp_y)
        
        
    f.close()
    
input_size = 42

x_train_origin = np.array(train_images)
x_train_origin = x_train_origin.reshape((28709,48,48))
y_train_origin = np.array(train_labes)
x_test_origin = np.array(test_images)
x_test_origin = x_test_origin.reshape((3589,48,48))
y_test_origin = np.array(test_labes)

# Training set
x_train = np.concatenate(
    (x_train_origin[:, 0:0+input_size, 0:0+input_size],
     x_train_origin[:, 0:0+input_size, 6:6+input_size],
     x_train_origin[:, 6:6+input_size, 0:0+input_size],
     x_train_origin[:, 6:6+input_size, 6:6+input_size],
     x_train_origin[:, 3:3+input_size, 3:3+input_size],)
)
x_train = np.concatenate((x_train, np.flip(x_train, 2)))
y_train = np.tile(y_train_origin, 10)

# Test set
x_test = np.concatenate(
    (x_test_origin[:, 0:0+input_size, 0:0+input_size],
     x_test_origin[:, 0:0+input_size, 6:6+input_size],
     x_test_origin[:, 6:6+input_size, 0:0+input_size],
     x_test_origin[:, 6:6+input_size, 6:6+input_size],
     x_test_origin[:, 3:3+input_size, 3:3+input_size],)
)
x_test = np.concatenate((x_test, np.flip(x_test, 2)))
y_test = np.tile(y_test_origin, 10)

x_train = x_train.reshape((28709*10,input_size,input_size,1))
x_test = x_test.reshape((3589*10,input_size,input_size,1))
x_train, x_test = x_train/225.0, x_test/225.0

# set gpu auto allocate memory
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)

model = models.Sequential([
    layers.Conv2D(32, (5,5), strides=1, padding='same', input_shape=(input_size,input_size,1), activation='relu'),
    layers.MaxPooling2D(pool_size=(3,3), strides=2),
    layers.Conv2D(32, (4,4), strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3,3), strides=2),
    layers.Conv2D(64, (5,5), strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3,3), strides=2),
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(7, activation='softmax')
])
# model.summary() # display model structure

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = './models/model1/model.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True)

# model.save_weights(checkpoint_path.format(epoch=0))
with tf.device('/gpu:0'):
    model.fit(x_train, y_train, batch_size=50, epochs=50, callbacks=[cp_callback])
    model.evaluate(x_test, y_test)

model_path = './models/model1/model.h5'
model.save(model_path)
