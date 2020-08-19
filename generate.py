import functions
from self_attention import *
import decode2
import numpy as np
#
# model = Transformer(num_layers=settings.num_layers,
#                     d_model=settings.embed_dim,
#                     num_heads=settings.num_heads,
#                     dff=settings.dense_layer_units,
#                     vocab_size=settings.vocab_size,
#                     pe_input=settings.seq_len)
# learning_rate = CustomSchedule(settings.embed_dim)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, settings.checkpoint_dir, max_to_keep=5)
#
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()  # since we'll load the optimizer as well
#     print('Latest checkpoint restored.')

model = tf.keras.models.load_model('Models/SmAll/fast3')
model.run_eagerly = True
# model.build(tf.TensorShape([1, None]))
# model.summary()
# dataSequence = DataSequence(settings.dataset_dir, settings.batch_size, settings.seq_len)
# test_x, test_y = dataSequence.get_data('test')
touhou = False
# print("Generating w/o starter...")
# no_starter = functions.generate(model, [322,  10,   4,  70, 283, 198], 3000, 'Liszt', temperature=0.7)
# # model.save('Models/SmAll/fast')
# decode2.decode(no_starter, 'test.mid', touhou=touhou)
print("Generating w/ starter...")
starter = functions.generate(model, np.load('Music/Beethoven/wordEvents/elise.mid.npy')[:100], 3000, 'Beethoven', temperature=1.0)
decode2.decode(starter, 'starter test.mid', touhou=touhou)
model.save('Models/SmAll/fast4')
# print("Generating w/ TOUHOU starter... uwu")
# touhou_gen = functions.generate(model, np.load('Touhou/touhou/wordEvents/sh01_01.mid0.npy')[:100], 3000, temperature=0.75)
# decode2.decode(touhou_gen, 'touhou.mid', touhou=touhou)
# model.summary()
# print("Generating w/ starter... using argmax...")
# starter_max = functions.generate(model, settings.starter, 3000, temperature=0.8, argmax=True)
# decode2.decode(starter_max, 'starter_max.mid')
# print("Generating w/o starter... using argmax...")                                            cc
# no_starter_max = functions.generate(model, [322, 10, 4], 3000, temperature=0.7, argmax=True)
# decode2.decode(no_starter_max, 'no_starter_max.mid')
# full_no_starter = functions.generate_full(model, tf.expand_dims(tf.constant(322), axis=0), temperature=0.7)
# decode2.decode(full_no_starter, 'full_no_starter.mid')
