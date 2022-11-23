# Multiclass Classification
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0


model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(x_train_normalized[..., np.newaxis], y_train, epochs=5)

model.evaluate(x_test_normalized[..., np.newaxis], y_test)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------


# Plant Disease Classification
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import math


(train, test), metadata = tfds.load('plant_village',
                                    split=['train', 'test'],
                                    as_supervised=True,
                                    with_info=True)

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train = train.map(normalize)
test = test.map(normalize)

BATCH_SIZE = 32

model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2), strides=2),
    layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(38, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train, epochs=5)

test_loss, test_accuracy = model.evaluate(test)
print('Accuracy on test dataset:', test_accuracy)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------


# Face Emotion Classification
# Human Activity Recognition
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# face emotion detection

data_dir = 'data/fer2013'

'''
data
    - train
        - class 1 folder
            - image 1
            - image 2
            - ...
        - class 2 folder
            - image 1
            - image 2
            - ...
        - ...
    - validation
        - class 1 folder
            - image 1
            - image 2
            - ...
        - class 2 folder
            - image 1
            - image 2
            - ...
        - ...

'''

train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(48, 48),                   
                                                    batch_size=64,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                target_size=(48, 48),           
                                                                batch_size=64,
                                                                color_mode='grayscale',
                                                                class_mode='categorical')

model = tf.keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_generator, epochs=15, validation_data=validation_generator)

model.save('model.h5')

