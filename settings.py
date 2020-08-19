import os
# dataset_dir = ['Music/Beethoven/combined-word-events', 'Music/Mozart/wordEvents', 'Music/Bach/wordEvents',
#                'Music/Chopin/wordEvents', 'Music/Haydn/wordEvents']
dataset_dir = [f.name for f in os.scandir('MUSIC') if f.is_dir()]
checkpoint_dir = 'Models/SmAll/256e1024h6l512s2b8h512d0.3dr'

# preprocessing etc.
sampling_freq = 12  # ticks per beat
pause_dim = sampling_freq * 4  # one measure of pause max
velocity_dim = 6  # velocity dim, pp-ff
min_tempo = 40
max_tempo = 200
tempo_step = 20
tempo_dim = (max_tempo-min_tempo) // tempo_step + 1
note_dim = 128 * 2
artist_dim = len(dataset_dir)  # is this a good idea?
vocab_size = velocity_dim + tempo_dim + note_dim + pause_dim + 4 + artist_dim + 1  # start, end, pedal on, pedal off
ticks_per_beat = 240
shifts = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

# dim offsets
velocity_offset = 0
tempo_offset = velocity_offset + velocity_dim
note_offset = tempo_offset + tempo_dim
pause_offset = note_offset + note_dim
pedal_offset = pause_offset + pause_dim
artist_offset = pedal_offset + 4

# training
embed_dim = 256
num_hid = 1024
num_layers = 6
epochs = 5000
seq_len = 512
batch_size = 2
num_heads = 8
dense_layer_units = 512

# getting data
train_split = 0.9
val_split = 0.05



starter = [322,  10,   4,  70, 283,  75, 272, 198, 283,  68, 272, 203, 280, 70, 272, 196, 274,  71, 272, 198,
           283,  65, 272, 199, 283,  70, 272, 193, 283,  198, 283]
