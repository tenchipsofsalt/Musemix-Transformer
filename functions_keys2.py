import os
import tensorflow as tf
import numpy as np
import settings_keys2
from self_attention_keys2 import create_masks
import time


# get files matching extension in directory without looking at subfolders
def get_files(dir_path, extension):
    files = []
    for (path, _, filenames) in os.walk(dir_path):
        files.extend(os.path.join(path, filename) for filename in filenames if filename.lower().endswith(extension))
        break
    return files


# generate given a data sequence, length, artist etc. Supports argmax (always picking most likely element) and top-k
# (picking from top k most likely elements) Artist must be in settings_keys2.dataset_dir
def generate(model, data, length, artist=None, temperature=1.0, argmax=False, k=None):
    original_data = np.array(data)
    key_event = tf.reshape(original_data[0], [1, 1])
    data = tf.expand_dims(data, 0)
    if artist is not None:
        try:
            artist_id = settings_keys2.artist_offset + settings_keys2.dataset_dir.index(artist) + 1
            print(artist_id)
            artist_id = tf.reshape(artist_id, [1, 1])
        except ValueError:
            print('Artist not found.')
            return
        data = tf.concat([artist_id, data], 1)
    cur_percent = 1
    # total_start = time.time()
    start = time.time()
    # last_tracked_time = total_start
    for i in range(length):
        # this doesn't work for sequences under 100, too lazy to fix atm... it's just a slight ETA problem.
        if i >= (length / 100) * cur_percent:
            print(f'{cur_percent}% done...')
            diff = time.time()-start
            print(f'That took {diff} seconds, ETA: {diff * (100-cur_percent)}')
            cur_percent += 1
            start = time.time()
        shape = data.shape[1]
        if shape >= settings_keys2.seq_len:
            shape = settings_keys2.seq_len  # caps it at this value
            if artist is not None:
                data = data[:, -(settings_keys2.seq_len - 2):]
                data = tf.concat([key_event, artist_id, data], 1)
            else:
                data = data[:, -(settings_keys2.seq_len - 1):]
                data = tf.concat([key_event, data], 1)
            temp_data = data
        else:
            paddings = tf.constant([[0, 0], [0, settings_keys2.seq_len-shape]])
            temp_data = tf.pad(data, paddings, "CONSTANT")
        print(temp_data)
        look_ahead_mask = create_masks(temp_data)
        # preprocess_time = time.time() - last_tracked_time
        # last_tracked_time += preprocess_time
        print(shape)
        predictions = model([temp_data, False, look_ahead_mask])

        # predict_time = time.time() - last_tracked_time
        # last_tracked_time += predict_time

        # get last element, remove batch dimension
        predictions = tf.squeeze(predictions[:, shape-1:shape, :], axis=0)
        if argmax:
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        else:
            # tried doing something with repeated notes and increasing temperature, not sure it had much of an effect
            # replace temperature and k in lines 91/95 with their temporary versions to try it
            # temp_temperature = temperature
            # scalar = 1.2
            # temp_k = k
            # try:
            #     max_repeated = np.amax(np.bincount(original_data[-40:])
            #                            [settings_keys2.note_offset + 1:settings_keys2.pause_offset + 1])
            #     if max_repeated >= 4:
            #         temp_temperature *= np.power(scalar, max_repeated - 3)
            #         # temp_k += (np.amax(np.bincount(original_data[-20:])) - 3)
            # except ValueError:
            #     pass

            # make repeats less likely
            # scalar = 1.2
            # predictions = predictions.numpy()
            # for element in generated[-48:]:
            #     if settings_keys2.pause_offset >= element > settings_keys2.note_offset:
            #         if predictions[0][element] < 0:
            #             predictions[0][element] *= scalar
            #         else:
            #             predictions[0][element] /= scalar

            predictions = predictions / temperature

            # top k
            if k is not None:
                values, indices = tf.math.top_k(predictions, k=k)
                predicted_id = indices[0][tf.cast(tf.random.categorical(values, num_samples=1)[-1], tf.int32)[0]]
            # everything
            else:
                predicted_id = tf.cast(tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy(), tf.int32)
            predicted_id = tf.expand_dims(predicted_id, axis=0)
        # add to input data
        data = tf.concat([data, tf.expand_dims(predicted_id, axis=0)], -1)
        # reduce length of input data if too long
        original_data = np.append(original_data, predicted_id.numpy())
        print(original_data)
        print(len(original_data))
        # if original_data[-1] == settings_keys2.vocab_size - 1:
        #     print('Found end token!')
        #     break
        # done_time = time.time() - last_tracked_time
        # last_tracked_time += done_time
    print(f'{cur_percent}% done...')
    return original_data
