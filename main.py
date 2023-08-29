import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

train_dir = "train"
validation_dir ="val"

#Constants

IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CLASSES = 7
EPOCHS = 50
FINE_TUNING_EPOCHS = 10
EARLY_STOPPING_CRITERIA = 3
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
feature_extractor = DenseNet169(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet")(inputs)

x = GlobalAveragePooling2D()(feature_extractor)
x = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
classification_output = Dense(NUM_CLASSES, activation="softmax", name="classification")(x)

model = Model(inputs=inputs, outputs=classification_output)
model.compile(
    optimizer=tf.keras.optimizers.SGD(0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.layers[1].trainable = False

earlyStoppingCallback = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_CRITERIA,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    x=train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[earlyStoppingCallback]
)

model.save("model_before_fine_tuning.h5")

model.layers[1].trainable = True

model.compile(
    optimizer=tf.keras.optimizers.SGD(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

earlyStoppingCallback_fine_tune = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_CRITERIA,
    verbose=1,
    restore_best_weights=True
)

history_fine_tune = model.fit(
    x=train_generator,
    epochs=FINE_TUNING_EPOCHS,
    validation_data=validation_generator,
    callbacks=[earlyStoppingCallback_fine_tune]
)

model.save("model_after_fine_tuning.h5")

