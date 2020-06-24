import csv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import numpy as np
import os
import nets

with open('fer2013.csv') as f:
    f_csv = csv.DictReader(f)
    
    # Initialize
    train_images = []
    train_labes = [] 
    test_images = []
    test_labes = [] 
    test_images_private = []
    test_labes_private = []
    
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
            
        elif usage == "PrivateTest":
            test_images_private.append(temp_x)
            test_labes_private.append(temp_y)
        
    f.close()

x_test = np.array(test_images)
x_test = x_test.reshape((3589,48,48,1))
y_test = np.array(test_labes)

x_test_private = np.array(test_images_private)
x_test_private = x_test_private.reshape((3589,48,48,1))
y_test_private = np.array(test_labes_private)

model = nets.getnet2()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path2 = "./models_3/model.ckpt"
checkpoint_dir2 = os.path.dirname(checkpoint_path2)
latest = tf.train.latest_checkpoint(checkpoint_dir2)

model.load_weights(latest)

with tf.device('/cpu:0'):
    model.evaluate(x_test, y_test)
