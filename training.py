
import csv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import numpy as np
import os
# from class_names import *

with open('fer2013.csv') as f:
    f_csv = csv.DictReader(f)
    
    # Initialize
    # i = 0
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
        # i+=1
        # if i>0:
        #     break

x_train = np.array(train_images)
x_train = x_train.reshape((28709,48,48))
y_train = np.array(train_labes)
x_test = np.array(test_images)
x_test = x_test.reshape(3589,48,48)
y_test = np.array(test_labes)

x_train, x_test = x_train / 255.0, x_test / 255.0

# print(x_train.shape)
# model = models.Sequential()
# model.add(layers.Conv2D)

model = models.Sequential([
    layers.Convolution1D(32, 5, strides=1, padding='same', input_shape=(48,48), activation='relu'),
    layers.MaxPooling1D(pool_size=3, strides=2),
    layers.Convolution1D(32, 4, strides=1, padding='same', activation='relu'),
    layers.MaxPooling1D(pool_size=3, strides=2),
    layers.Convolution1D(32, 5, strides=1, padding='same', activation='relu'),
    layers.MaxPooling1D(pool_size=3, strides=2),
    layers.Flatten(),
    layers.Dense(2048,activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1024,activation='softmax'),
    layers.Dropout(0.4)
])

# model.summary()

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


checkpoint_path = "./models/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=5)

# model.save_weights(checkpoint_path.format(epoch=0))
with tf.device('/cpu:0'):
    model.fit(x_train, y_train, epochs=1500, callbacks=[cp_callback])
    model.evaluate(x_test, y_test)





