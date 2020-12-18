import functions_keys2
from self_attention_keys2 import *
import decode_keys2
import numpy as np

# define architecture
model = Transformer(num_layers=settings_keys2.num_layers,
                    d_model=settings_keys2.embed_dim,
                    num_heads=settings_keys2.num_heads,
                    dff=settings_keys2.dense_layer_units,
                    vocab_size=settings_keys2.vocab_size,
                    pe_input=settings_keys2.seq_len)
learning_rate = CustomSchedule(settings_keys2.embed_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, settings_keys2.checkpoint_dir, max_to_keep=5)

# load weights
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()  # since we'll load the optimizer as well
    print('Latest checkpoint restored.')

# # load savedModel format
# model = tf.keras.models.load_model('Models/SmAll/fast3')
# model.run_eagerly = True

# just some midi header info
touhou = False

# generate
print("Generating w/ starter...")
starter = functions_keys2.generate(model, np.load('Music/Bach/keyedEvents2/17 Polonaise.mid.npy')[:512], 1000, artist="Albéniz", temperature=1.0)
decode_keys2.decode(starter, 'keyedblanktest.mid', touhou=touhou)
# model.save('INSERT PATH HERE')
