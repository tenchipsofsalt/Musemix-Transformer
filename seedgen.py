import functions
import random
import settings
import numpy as np

dirs = settings.dataset_dir
count = settings.artist_offset
for directory in dirs:
    count += 1
    files = functions.get_files(f'Music/{directory}/wordEvents', '.npy')
    file = random.choice(files)
    filename = file.split('\\')[-1].split('.mid')[0].translate({ord(i): None for i in " .',()"})
    data = np.load(file)[:99]
    f = open(f'seeds/{directory}{filename}.txt', 'x')
    data_str = f"{count}, "
    for elem in data[:-1]:
        data_str += str(elem) + ', '
    data_str += str(data[-1])
    f.write(data_str)
    f.close()
