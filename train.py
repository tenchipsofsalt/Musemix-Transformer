import os
from dataset import *
import functions


def loss(labels, logits):
    scalar = 1.5
    max_look_back = 48
    # could've done this with a python function, ig I'm dumb
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    last_logits = tf.transpose(logits, perm=(1, 0, 2))[-1]
    predictions = tf.map_fn(elems=last_logits, fn=lambda t: tf.cast(tf.math.argmax(t, 0), tf.int32), dtype=tf.int32)
    look_back = 0-tf.math.minimum(tf.shape(labels)[-1], max_look_back)  # look back n elements or less if does not exist
    transposed_actual = tf.cast(tf.transpose(labels, perm=(1, 0))[look_back:-1], tf.int32)
    matches = tf.map_fn(elems=transposed_actual,
                        fn=lambda t:
                        tf.where(tf.logical_and(tf.logical_and(tf.equal(t, predictions),
                                                               tf.less_equal(predictions, settings.pause_offset)),
                                                tf.greater(predictions, settings.note_offset)), scalar, 1),
                        dtype=tf.float32)
    penalty = tf.reduce_mean(tf.reduce_prod(matches, axis=0))
    return loss_value * penalty


# from my understanding:
# num_token = vocab size
# embed_dim = embedding dimension
# num_hid = number of hidden units
# num_layers = number of LSTM layers
# pass in [batch_size, seq_len]
num_token = settings.vocab_size
embed_dim = settings.embed_dim
num_hid = settings.num_hid
num_layers = settings.num_layers
checkpoint_dir = settings.checkpoint_dir
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True,
                                                         save_best_only=True)


# load dataset
dataSequence = DataSequence(settings.dataset_dir, settings.batch_size, settings.seq_len)
# OOM, not feasible
# train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_all(settings.seq_len)

val_x, val_y = dataSequence.get_data('val')
test_x, test_y = dataSequence.get_data('test')

model = functions.build_model(vocab_size=num_token, embedding_dim=embed_dim, rnn_units=num_hid,
                              batch_size=settings.batch_size)

try:
    pass
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
except AttributeError:
    print('No checkpoint found... continuing...')
except ValueError:
    print('Invalid checkpoint found... continuing...')
model.compile(optimizer='rmsprop', loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
history = model.fit(x=dataSequence, batch_size=2, epochs=settings.epochs, callbacks=[checkpoint_callback, lr],
                    validation_data=(val_x, val_y))



