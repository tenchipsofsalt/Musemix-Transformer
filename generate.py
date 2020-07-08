import tensorflow as tf
import settings
import functions
from dataset import DataSequence
import decode2

model = functions.build_model(settings.vocab_size, settings.embed_dim, settings.num_hid, batch_size=1, training=False)
model.load_weights(tf.train.latest_checkpoint(settings.checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# dataSequence = DataSequence(settings.dataset_dir, settings.batch_size, settings.seq_len)
# test_x, test_y = dataSequence.get_data('test')

print("Generating w/ starter...")
starter = functions.generate(model, settings.starter, 3000, temperature=0.65)
decode2.decode(starter, 'starter.mid')
print("Generating w/o starter...")
no_starter = functions.generate(model, tf.expand_dims(tf.constant(322), axis=0), 3000, temperature=0.7)
decode2.decode(no_starter, 'no_starter.mid')
# print("Generating w/ starter... using argmax...")
# starter_max = functions.generate(model, settings.starter, 3000, temperature=0.65, argmax=True)
# decode2.decode(starter_max, 'starter_max.mid')
# print("Generating w/o starter... using argmax...")
# no_starter_max = functions.generate(model, tf.expand_dims(tf.constant(322), axis=0), 3000, temperature=0.7, argmax=True)
# decode2.decode(no_starter_max, 'no_starter_max.mid')
# full_no_starter = functions.generate_full(model, tf.expand_dims(tf.constant(322), axis=0), temperature=0.7)
# decode2.decode(full_no_starter, 'full_no_starter.mid')
