import functions
from self_attention import *
import numpy as np
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()  # since we'll load the optimizer as well
    print('Latest checkpoint restored.')

# run some data through to init
starter = functions.generate(model, np.load('Music/Beethoven/wordEvents/elise.mid.npy')[:100], 10, 'Beethoven')

artist_embeddings = model.layers[0].embedding.get_weights()[0][324:]

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=1000)
tsne_results = tsne.fit_transform(artist_embeddings)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(16, 10))
x = tsne_results[:, 0]
y = tsne_results[:, 1]
ax = sns.scatterplot(
    x=x, y=y,
    color='#d61347',
    legend=False,
    alpha=0.7
)
for i, value in enumerate(settings.dataset_dir):
    ax.annotate(value, (x[i] + 3, y[i] + 3))
ax.axis('off')
plt.savefig('embedding distrib.png', transparent=True)
plt.show()
