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
SUFFIX = "baseline"
TARGET_RESOLUTION = 128
BATCH_SIZE = 32
EPOCHS = 1
FINE_TUNE_EPOCHS = 80


def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1+keras.backend.epsilon())) - tf.reduce_mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + keras.backend.epsilon()))
	return focal_loss_fixed

train_image_generator_def = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
)
eval_image_generator_def = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
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
        [None, TARGET_RESOLUTION, TARGET_RESOLUTION, 3],
        [None,]
    ),
)
train_gen = train_gen.map(lambda images, labels: (images, labels, labels * 1.25 + 1))

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
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.elu(x)
x = keras.layers.Dropout(0.1)(x)
x = keras.layers.Dense(256)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.elu(x)
x = keras.layers.Dropout(0.1)(x)
x = keras.layers.Dense(128, activation="leaky_relu")(x)
x = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.models.Model(inputs=model_base.inputs, outputs=x)
model.summary()
model.compile(
    optimizer="adam",
    loss=focal_loss(),
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_{SUFFIX}.csv")
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping_callback, csv_logger],
    steps_per_epoch = len(train_gen_base)
)

# fine tune all
for layer in model.layers:
    layer.trainable = True

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_{SUFFIX}_test.csv"), append=True
)
model.compile(
    optimizer="adam",
    loss=focal_loss(),
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stopping_callback, csv_logger],
    steps_per_epoch = len(train_gen_base)
)
model.evaluate(test_gen)
model.save(os.path.join(OUTPUT_PATH, MODEL_NAME), include_optimizer=False)
