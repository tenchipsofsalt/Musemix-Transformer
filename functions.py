import os
import tensorflow as tf
import numpy as np
import settings


def get_files(dir_path, extension):
    files = []
    for (path, _, filenames) in os.walk(dir_path):
        files.extend(os.path.join(path, filename) for filename in filenames if filename.endswith(extension))
        break
    return files


# Adapted from https://www.tensorflow.org/tutorials/text/text_generation#generate_text
def build_model(vocab_size, embedding_dim, rnn_units, batch_size, training=True):
    model = tf.keras.Sequential([])
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]))
    model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(vocab_size))
    # built = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    #     tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    #     tf.keras.layers.Dropout(0.4),
    #     tf.keras.layers.Dense(vocab_size)
    # ])
    return model


# Adapted from https://www.tensorflow.org/tutorials/text/text_generation#generate_text
def generate(model, data, length, temperature=1.0, argmax=False):
    original_data = np.array(data)
    data = tf.expand_dims(data, 0)

    generated = []

    model.reset_states()
    for i in range(length):
        predictions = model(data)

        # removing batch dimension
        predictions = tf.squeeze(predictions, 0)

        if argmax:
            predicted_id = np.argmax(predictions[-1])
        else:
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        data = tf.expand_dims([predicted_id], 0)

        generated.append(predicted_id)

    return np.append(original_data, generated)


def generate_full(model, data, temperature=1.0):
    original_data = np.array(data)
    data = tf.expand_dims(data, 0)

    generated = []

    model.reset_states()
    predictions = model(data)

    # removing batch dimension
    predictions = tf.squeeze(predictions, 0)

    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

    data = tf.expand_dims([predicted_id], 0)

    generated.append(predicted_id)
    while generated[-1] != settings.pedal_offset + 4:
        print(f'Length: {len(generated)}')
        predictions = model(data)

        # removing batch dimension
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        data = tf.expand_dims([predicted_id], 0)

        generated.append(predicted_id)

    return np.append(original_data, generated)
