import functions
from self_attention import *
import decode2
import numpy as np

# define architecture
model = Transformer(num_layers=settings.num_layers,
                    d_model=settings.embed_dim,
                    num_heads=settings.num_heads,
                    dff=settings.dense_layer_units,
                    vocab_size=settings.vocab_size,
                    pe_input=settings.seq_len)
learning_rate = CustomSchedule(settings.embed_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, settings.checkpoint_dir, max_to_keep=5)

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
starter = functions.generate(model, np.load('Music/Albéniz/wordEvents/Aragon (Fantasia) Op.47 part 6.mid.npy')[:100], 3000)
decode2.decode(starter, 'starter test.mid', touhou=touhou)
# model.save('INSERT PATH HERE')
