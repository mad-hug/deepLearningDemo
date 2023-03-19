import os

import tensorflow as tf
from keras import Model
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def demo_cnn():
    image_size = (32, 768)
    batch_size = 32
    num_classes = 3
    epochs = 10
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'TTW(dataset)\\train(576)',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'TTW(dataset)\\val(144)',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    resnet_model = ResNet50(include_top=False, weights='imagenet')

    for layer in resnet_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        resnet_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    end_time = time.time()

    print("Training time: {:.2f} seconds".format(end_time - start_time))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training loss')
    plt.plot(epochs_range, val_loss, label='Validation loss')
    plt.legend(loc='lower right')
    plt.title('Training and Validation loss')
    plt.show()

    loss, accuracy = model.evaluate(val_ds)
    print("Validation accuracy: {:.2f}%".format(accuracy * 100))

    y_predict_prob = model.predict(val_ds)
    y_predict = np.argmax(y_predict_prob, axis=1)
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    accuracy = np.mean(y_predict == y_true) * 100
    print("Test accuracy: {:.2f}%".format(accuracy))
    cm = confusion_matrix(y_true, y_predict)
    plt.figure(figsize=(6, 6))
    class_names = os.listdir('TTW(dataset)\\train(576)')
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
