
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  
  layers.Convolution1D(32, 5, strides=1, padding='same', input_shape=(28,28), activation='relu'),
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


with tf.device('/cpu:0'):
    model.fit(x_train, y_train, batch_size=10, epochs=5)
  
model.evaluate(x_test, y_test)
