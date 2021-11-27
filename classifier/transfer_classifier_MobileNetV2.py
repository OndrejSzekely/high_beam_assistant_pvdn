import os
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

TRAINING_IMAGES_FOLDER_PATH = "/digiteq/prepared_clasification_dataset/train"
VALIDATION_IMAGES_FOLDER_PATH = "/digiteq/prepared_clasification_dataset/val"
TESTING_IMAGES_FOLDER_PATH = "/digiteq/prepared_clasification_dataset/test"
OUTPUT_PATH = "/digiteq/models"
MODEL_NAME = "mobilenet_v2"
SUFFIX = "baseline"
TARGET_RESOLUTION = 71
BATCH_SIZE = 32
EPOCHS = 30

train_image_generator_def = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
)
eval_image_generator_def = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
)

train_gen_base = train_image_generator_def.flow_from_directory(
    TRAINING_IMAGES_FOLDER_PATH,
    target_size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
)

train_gen = tf.data.Dataset.from_generator(
    lambda: train_gen_base,
    output_types=(tf.float32, tf.float32),
    output_shapes=(
        [BATCH_SIZE, TARGET_RESOLUTION, TARGET_RESOLUTION, 3],
        [BATCH_SIZE,]
    ),
)
train_gen = train_gen.map(lambda images, labels: (images, labels, labels * 2.5 + 1))

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

model_base = keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(TARGET_RESOLUTION, TARGET_RESOLUTION, 3),
    pooling="avg",
)

x = model_base.output
x = keras.layers.Flatten()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(512, activation="leaky_relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(256, activation="leaky_relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(1, activation="sigmoid")(x)

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
csv_logger_eval = tf.keras.callbacks.CSVLogger(
    os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_{SUFFIX}_test.csv")
)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping_callback, csv_logger_train],
    steps_per_epoch = len(train_gen_base)
)
model.evaluate(test_gen, callbacks=[csv_logger_eval])
model.save(os.path.join(OUTPUT_PATH, MODEL_NAME), include_optimizer=False)
