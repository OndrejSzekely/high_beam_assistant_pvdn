import os
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

TRAINING_IMAGES_FOLDER_PATH = "/digiteq/prepared_clasification_dataset/train"
VALIDATION_IMAGES_FOLDER_PATH = "/digiteq/prepared_clasification_dataset/val"
TESTING_IMAGES_FOLDER_PATH = "/digiteq/prepared_clasification_dataset/test"
OUTPUT_PATH = "/digiteq/models"
MODEL_NAME = "xception"
SUFFIX = "fine_tuned"
TARGET_RESOLUTION = 71
BATCH_SIZE = 64
EPOCHS = 30
FINE_TUNE_EPOCHS = 10

train_image_generator_def = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
)
eval_image_generator_def = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
)

train_gen = train_image_generator_def.flow_from_directory(
    TRAINING_IMAGES_FOLDER_PATH,
    target_size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
)

val_gen = eval_image_generator_def.flow_from_directory(
    VALIDATION_IMAGES_FOLDER_PATH,
    target_size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
)

test_gen = eval_image_generator_def.flow_from_directory(
    TESTING_IMAGES_FOLDER_PATH,
    target_size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
)

model_base = keras.applications.Xception(
    include_top=False,
    input_shape=(TARGET_RESOLUTION, TARGET_RESOLUTION, 3),
    pooling="avg",
)

for layer in model_base.layers:
    layer.trainable = False

x = model_base.output
x = keras.layers.Flatten(name="custom/flatten_1")(x)
x = keras.layers.BatchNormalization(name="custom/batchnorm_1")(x)
x = keras.layers.Dense(512, activation="leaky_relu", name="custom/dense_1")(x)
x = keras.layers.BatchNormalization(name="custom/batchnorm_2")(x)
x = keras.layers.Dense(256, activation="leaky_relu", name="custom/dense_2")(x)
x = keras.layers.BatchNormalization(name="custom/batchnorm_3")(x)
x = keras.layers.Dense(1, activation="sigmoid", name="custom/dense_3")(x)

model = keras.models.Model(inputs=model_base.inputs, outputs=x)
model.summary()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True
)
csv_logger_train = tf.keras.callbacks.CSVLogger(
    os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_{SUFFIX}.csv")
)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping_callback, csv_logger_train],
)

# fine tune all
for layer in model.layers:
    layer.trainable = True
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True
)
csv_logger_train = tf.keras.callbacks.CSVLogger(
    os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_{SUFFIX}.csv"), append=True
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stopping_callback, csv_logger_train],
)

model.evaluate(test_gen)
model.save(os.path.join(OUTPUT_PATH, MODEL_NAME), include_optimizer=False)
