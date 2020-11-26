import os

dataset_dir = [f.name for f in os.scandir('Music') if f.is_dir()]
checkpoint_dir = 'Models/keyed/8e512h6l512s4b8h256d0.3drno_artist/'

# preprocessing etc.
sampling_freq = 12  # ticks per beat
pause_dim = sampling_freq * 4  # one measure of pause max
velocity_dim = 6  # velocity dim, pp-ff
min_tempo = 40
max_tempo = 200
tempo_step = 20
tempo_dim = (max_tempo-min_tempo) // tempo_step + 1
note_dim = 128 * 2 * 24
artist_dim = len(dataset_dir)
vocab_size = velocity_dim + tempo_dim + note_dim + pause_dim + 4 + artist_dim + 1  # 4 = start, end, pedal on, pedal off
ticks_per_beat = 240
shifts = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

# dim offsets, exclusive on bottom, inclusive on top
velocity_offset = 0
tempo_offset = velocity_offset + velocity_dim
note_offset = tempo_offset + tempo_dim
pause_offset = note_offset + note_dim
pedal_offset = pause_offset + pause_dim
artist_offset = pedal_offset + 4

# training
embed_dim = 8
num_hid = 512
num_layers = 6
epochs = 5000
seq_len = 512
batch_size = 4
num_heads = 8
dense_layer_units = 256

# getting data
train_split = 0.9
val_split = 0.05
