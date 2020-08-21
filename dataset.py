import numpy as np
import decode2
import functions
import random
import tensorflow as tf
import math
import settings
import preprocess3


# didn't really implement this fully, maybe consider in future with a generate function as well
class SongData:
    def __init__(self, file):
        self.events = np.load(file)

    def decode(self, output_name):
        decode2.decode(self.events, output_name)


# data pipeline
# __getitem__, or indexing will return a randomly shifted sequence from the corresponding data point,
# with artist id first
# get_data basically repeats that for a whole part of the data (train, val, or test)
class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, dir_names, batch_size, seq_len, train=settings.train_split, val=settings.val_split):
        self.files = []

        # artists
        count = 0
        for dir_name in dir_names:
            count += 1
            for file in functions.get_files(f'Music/{dir_name}/wordEvents/', '.npy'):
                self.files.append([file, count])
        random.shuffle(self.files)
        self.file_dict = {
            'train': self.files[:int(len(self.files) * train)],
            'val': self.files[int(len(self.files) * train):int(len(self.files) * (train + val))],
            'test': self.files[int(len(self.files) * (train + val)):],
        }
        self.batch_size = batch_size
        self.seq_len = seq_len

        # check all files are long enough
        for file in self.files:
            if len(np.load(file[0])) <= seq_len:
                print(f'File {file} is too short. Please remove and try again.')

    def __len__(self):
        return math.ceil(len(self.file_dict['train']) / self.batch_size)

    def __getitem__(self, idx, source='train'):
        if idx == 0:
            random.shuffle(self.file_dict[source])
        batch = self.file_dict[source][idx * self.batch_size:(idx + 1) * self.batch_size]
        seqs = []
        for file in batch:
            seqs.append(preprocess3.note_shift(_get_seq(file, self.seq_len + 1), random.choice(settings.shifts)))
        return [np.delete(seq, -1) for seq in seqs], [np.delete(seq, 1) for seq in seqs]

    def get_data(self, source):
        files = self.file_dict[source]
        seqs = []
        for file in files:
            seqs.append(_get_seq(file, self.seq_len + 1))
        while len(seqs) % self.batch_size != 0:
            seqs.pop()
        print(f'{source} data has length {len(seqs)}')
        return tf.convert_to_tensor([np.delete(seq, -1) for seq in seqs]), \
               tf.convert_to_tensor([np.delete(seq, 1) for seq in seqs])


# also not really implemented, takes too much memory to return everything
class Dataset:
    def __init__(self, dir_path, train=settings.train_split, val=settings.val_split):
        self.files = functions.get_files(dir_path, '.npy')
        self.file_dict = {
            'train': self.files[:int(len(self.files) * train)],
            'val': self.files[int(len(self.files) * train):int(len(self.files) * (train + val))],
            'test': self.files[int(len(self.files) * (train + val)):],
        }

    def get_batch(self, batch_size, seq_len, batch_type='train'):
        batch_files = random.sample(self.file_dict[batch_type], k=batch_size)
        batch_data = [_get_seq(file, seq_len) for file in batch_files]

        while None in batch_data:  # self explanatory, keeps pulling until you find a good batch ig
            print("Batch has a file that is too short, pulling new batch...")
            batch_files = random.sample(self.file_dict[batch_type], k=batch_size)
            batch_data = [_get_seq(file, seq_len) for file in batch_files]
        return np.array(batch_data)  # batch_size, seq_len

    def get_all(self, seq_len, step=1):
        train_x, train_y = get_all_of_type(self.file_dict, 'train', seq_len, step)
        val_x, val_y = get_all_of_type(self.file_dict, 'val', seq_len, step)
        test_x, test_y = get_all_of_type(self.file_dict, 'test', seq_len, step)
        return train_x, train_y, val_x, val_y, test_x, test_y


# Gets a random sequence from a file.
def _get_seq(file, seq_len):
    data = np.load(file[0])
    data_length = len(data)
    if seq_len > data_length:
        # Should be a zero-padding thing here. I didn't use it because almost all of my data was long enough for my
        # sequence length, which was limited by my computer's specs. I just cut out the few files that were too short.
        return None
    else:
        start = random.randrange(0, data_length - seq_len + 1)
        seq = data[start:start + seq_len - 1]
    seq = np.append([file[1] + settings.artist_offset], seq)
    return seq


# Gets ALL possible sequences of a certain length from file dict, which usually makes OOM errors.
# By all I mean 1-512, 2-513, 3-514 etc.
def get_all_of_type(file_dict, batch_type, seq_len, step=1, batch_size=1):
    x = []
    y = []
    for file in file_dict[batch_type]:
        cur_data = np.load(file)
        while len(cur_data) > seq_len + 1:
            x.append(cur_data[:seq_len])
            y.append(cur_data[1:seq_len + 1])  # if seq2seq, use [1:seq_len+1]
            cur_data = np.delete(cur_data, range(step))
    while len(x) % batch_size != 0:  # for validation, this needs to be a multiple of batch_size pepeHands
        x.pop()
        y.pop()
    x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
    print(f'{batch_type} data: {len(x)} elements of shape {x.shape}.')
    return x, y
