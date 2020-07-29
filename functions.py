import os
import tensorflow as tf
import numpy as np
import settings
from self_attention import create_masks
import time


def get_files(dir_path, extension):
    files = []
    for (path, _, filenames) in os.walk(dir_path):
        files.extend(os.path.join(path, filename) for filename in filenames if filename.lower().endswith(extension))
        break
    return files


def generate(model, data, length, artist, temperature=1.0, argmax=False, k=None):
    original_data = np.array(data)
    data = tf.expand_dims(data, 0)

    try:
        artist_id = settings.artist_offset + settings.dataset_dir.index(artist) + 1
        artist_id = tf.reshape(artist_id, [1, 1])
    except ValueError:
        print('Artist not found.')
        return
    cur_percent = 1
    total_start = time.time()
    start = time.time()
    # last_tracked_time = total_start
    data = tf.concat([artist_id, data], axis=1)
    for i in range(length):
        if i >= (length / 100) * cur_percent:
            print(f'{cur_percent}% done...')
            diff = time.time()-start
            print(f'That took {diff} seconds, ETA: {diff * (100-cur_percent)}')
            cur_percent += 1
            start = time.time()
        if data.shape[1] > settings.seq_len - 1:
            data = data[:, -(settings.seq_len - 1):]
            data = tf.concat([artist_id, data], axis=1)
        look_ahead_mask, dec_mask = create_masks(data)

        # preprocess_time = time.time() - last_tracked_time
        # last_tracked_time += preprocess_time
        # print(preprocess_time)

        @tf.function(input_signature=[tf.TensorSpec([1, None], tf.int32),
                                      tf.TensorSpec([], tf.bool),
                                      tf.TensorSpec([None, None], tf.float32)])
        def model_predict(d, train, mask):
            return model([d, train, mask])  # , dec_mask])

        predictions, weights = model_predict(data, False, look_ahead_mask)
        # predictions, weights = model([data, False, look_ahead_mask])

        # predict_time = time.time() - last_tracked_time
        # last_tracked_time += predict_time
        # print(predict_time)

        # get last element, remove batch dimension
        predictions = tf.squeeze(predictions[:, -1:, :], axis=0)
        if argmax:
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        else:
            temp_temperature = temperature
            scalar = 1.2
            temp_k = k
            try:
                max_repeated = np.amax(np.bincount(original_data[-40:])[settings.note_offset + 1:settings.pause_offset + 1])
                if max_repeated >= 4:
                    temp_temperature *= np.power(scalar, max_repeated - 3)
                    # temp_k += (np.amax(np.bincount(original_data[-20:])) - 3)
            except ValueError:
                pass
            predictions = predictions / temp_temperature

            # make repeats less likely
            # scalar = 1.2
            # predictions = predictions.numpy()
            # for element in generated[-48:]:
            #     if settings.pause_offset >= element > settings.note_offset:
            #         if predictions[0][element] < 0:
            #             predictions[0][element] *= scalar
            #         else:
            #             predictions[0][element] /= scalar

            # top k
            if k is not None:
                values, indices = tf.math.top_k(predictions, k=temp_k)
                predicted_id = indices[0][tf.cast(tf.random.categorical(values, num_samples=1)[-1], tf.int32)[0]]
            # everything
            else:
                predicted_id = tf.cast(tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy(), tf.int32)
        # add to input data
        # can use tf.reshape with newaxis?
        data = tf.concat([data, tf.expand_dims(tf.expand_dims(predicted_id, axis=0), axis=0)], axis=-1)
        # reduce length of input data if too long
        original_data = np.append(original_data, predicted_id.numpy())
        if original_data[-1] == settings.vocab_size - 1:
            print('Found end token!')
            break
        # done_time = time.time() - last_tracked_time
        # last_tracked_time += done_time
        # print(done_time)
    print(f'{cur_percent}% done...')
    return original_data


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
