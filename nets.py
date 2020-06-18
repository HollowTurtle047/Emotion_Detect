
from tensorflow.keras import layers, models

def getnet1():
    model = models.Sequential([
        layers.Convolution1D(32, 5, strides=1, padding='same', input_shape=(48,48), activation='relu'),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Convolution1D(32, 4, strides=1, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Convolution1D(64, 5, strides=1, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Flatten(),
        layers.Dense(2048,activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1024,activation='softmax'),
        layers.Dropout(0.2)
    ])
    return model

def getnet2():
    model = models.Sequential([
        layers.Conv2D(32, (5,5), strides=1, padding='same', input_shape=(48,48,1), activation='relu'),
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
    return model