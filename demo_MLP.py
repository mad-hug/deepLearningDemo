import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt


def demo_mlp(image_size=(32, 768), batch_size=32, num_classes=3, epochs=10):
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

    # Normalize input data
    normalization_layer = keras.layers.Rescaling(1. / 255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Define model architecture
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    start_time = time.time()
    history = model.fit(normalized_train_ds, epochs=epochs, validation_data=normalized_val_ds)
    end_time = time.time()

    # Print training time
    print("Training time: {:.2f} seconds".format(end_time - start_time))

    # Plot accuracy over time
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()

    # Evaluate model on validation data
    loss, accuracy = model.evaluate(normalized_val_ds)
    print("Validation accuracy: {:.2f}%".format(accuracy * 100))
