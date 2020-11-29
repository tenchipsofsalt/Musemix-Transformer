import time
from dataset_keys2 import *
from self_attention_keys2 import create_masks, Transformer, CustomSchedule
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if not os.path.exists(settings_keys2.checkpoint_dir):
    os.makedirs(settings_keys2.checkpoint_dir)

def loss_fn(labels, logits):
    mask = tf.math.logical_not(tf.math.equal(labels, 0))
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    mask = tf.cast(mask, loss_value.dtype)
    loss_value *= mask
    loss_value = tf.reduce_sum(loss_value)/tf.reduce_sum(mask)

    # could've done this with a python function, ig I'm dumb
    # testing out increasing loss based on repeated notes, don't think this is a good way to do it
    # scalar = 1.5
    # max_look_back = 48
    # last_logits = tf.transpose(logits, perm=(1, 0, 2))[-1]
    # predictions = tf.map_fn(elems=last_logits, fn=lambda t: tf.cast(tf.math.argmax(t, 0), tf.int32), dtype=tf.int32)
    # look_back = 0-tf.math.minimum(tf.shape(labels)[-1], max_look_back)  # look back n elems or less if does not exist
    # transposed_actual = tf.cast(tf.transpose(labels, perm=(1, 0))[look_back:-1], tf.int32)
    # matches = tf.map_fn(elems=transposed_actual,
    #                     fn=lambda t:
    #                     tf.where(tf.logical_and(tf.logical_and(tf.equal(t, predictions),
    #                                                            tf.less_equal(predictions, settings_keys2.pause_offset)),
    #                                             tf.greater(predictions, settings_keys2.note_offset)), scalar, 1),
    #                     dtype=tf.float32)
    # penalty = tf.reduce_mean(tf.reduce_prod(matches, axis=0))
    return loss_value  # * penalty


# from my understanding:
# num_token = vocab size
# embed_dim = embedding dimension
# num_hid = number of hidden units
# num_layers = number of LSTM layers
# pass in [batch_size, seq_len]
vocab_size = settings_keys2.vocab_size
embed_dim = settings_keys2.embed_dim
num_hid = settings_keys2.num_hid # doesn't do anything lol...
num_layers = settings_keys2.num_layers
checkpoint_dir = settings_keys2.checkpoint_dir
num_heads = settings_keys2.num_heads
dense_units = settings_keys2.dense_layer_units
epochs = settings_keys2.epochs
seq_len = settings_keys2.seq_len
batch_size = settings_keys2.batch_size

# load dataset
dataSequence = DataSequence(settings_keys2.dataset_dir, batch_size, seq_len)
# OOM, not feasible
# train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_all(settings_keys2.seq_len)

val_x, val_y = dataSequence.get_data('val')
test_x, test_y = dataSequence.get_data('test')

model = Transformer(num_layers=num_layers,
                    d_model=embed_dim,
                    num_heads=num_heads,
                    dff=dense_units,
                    vocab_size=vocab_size,
                    pe_input=seq_len)
learning_rate = CustomSchedule(embed_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_step_signature = [(tf.TensorSpec(shape=(batch_size, seq_len), dtype=tf.int32),
                         tf.TensorSpec(shape=(batch_size, seq_len), dtype=tf.int32))]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')


@tf.function(input_signature=train_step_signature)
def train_step(data):
    inp, target = data
    look_ahead_mask = create_masks(inp)

    with tf.GradientTape() as tape:
        test_pred = model([inp, True, look_ahead_mask])  # , dec_padding_mask])
        test_loss = loss_fn(target, test_pred)

    gradients = tape.gradient(test_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(test_loss)
    train_accuracy(target, test_pred)


print('Starting training...')
print(f'Test data has {batch_size * len(dataSequence)} elements, from {len(settings_keys2.dataset_dir)} sources.')

for epoch in range(epochs):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    for i in range(len(dataSequence)-1): # for some reason needs -1, too lazy to debug, shuffled so it's fine
        train_step(dataSequence[i])

        if i % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, i, train_loss.result(), train_accuracy.result()))

    for i in range(len(val_x)//batch_size):
        val_x_batch = val_x[i * batch_size:(i + 1) * batch_size]
        val_y_batch = val_y[i * batch_size:(i + 1) * batch_size]
        val_la_mask = create_masks(val_x_batch)
        val_pred = model([val_x_batch, False, val_la_mask])
        val_batch_loss = loss_fn(val_y_batch, val_pred)

        val_loss(val_batch_loss)
        val_accuracy(val_y_batch, val_pred)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
                                                    

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Epoch {} (Val) Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, val_loss.result(), val_accuracy.result()))
    f = open(checkpoint_dir + "log.txt", "a")
    # epoch, loss, accuracy, val loss, val acc
    f.write(f"{epoch + 1}, {train_loss.result()}, {train_accuracy.result()}, {val_loss.result()}, {val_accuracy.result()}\n")
    f.close()
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
