import mido
import settings
import functions


dirs = settings.dataset_dir
count = 0
length = 0
for directory in dirs:
    print(directory)
    files = functions.get_files(f'Music/{directory}/', '.mid')
    for file in files:
        try:
            length += mido.MidiFile(file).length
            count += 1
        except:
            pass

print(count)
print(length)
