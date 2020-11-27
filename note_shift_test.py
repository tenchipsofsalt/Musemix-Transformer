import preprocess_keys
import decode
import dataset
import numpy as np
temp = np.load('Music/Albéniz/wordEvents/Aragon (Fantasia) Op.47 part 6.mid.npy')
dataset.note_shift(temp, 0)
decode.decode(temp, 'shift_test_base.mid')