import tensorflow as tf
from tensorflow import keras


def demo_mlp(image_size=(32, 768), batch_size=32, num_classes=3, epochs=10):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'TTW(dataset)\\train(576)',
        image_size=(32, 768),
        batch_size=32,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'TTW(dataset)\\val(144)',
        image_size=(32, 768),
        batch_size=32,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    normalization_layer = keras.layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

#    # Build the model
#    model = tf.keras.Sequential([
#        keras.layers.experimental.preprocessing.Rescaling(1. / 255),
#        keras.layers.Flatten(input_shape=(32, 768, 3)),
#        keras.layers.Dense(num_classes, activation='softmax')
#    ])
#
#    # Compile the model
#    model.compile(optimizer='adam',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
#
#    # Train the model
#    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
#
#    # Evaluate the model on the validation set
#    loss, accuracy = model.evaluate(val_ds)
#
#    print('Validation accuracy:', accuracy)
