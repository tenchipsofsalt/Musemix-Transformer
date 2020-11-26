import music21
import functions
import settings
from os import path, makedirs



keydict = {}
dirs = settings.dataset_dir
for directory in dirs:
    files = functions.get_files(f'Music/{directory}/', '.mid')
    count = 0
    for file in files:
        try:
            midi = music21.converter.parse(file)
        except:
            print(file)
            continue
        key = midi.analyze('key').tonicPitchNameWithCase
        if key not in keydict:
            keydict[key] = 1
        else:
            keydict[key] += 1
        count += 1
        print(f'Done with file {file} Progress: {count} / {len(files)}.')
#{'F': 109, 'F#': 16, 'E-': 55, 'd': 97, 'D': 122, 'g': 69, 'E': 50, 'a': 70, 'C': 160, 'f#': 23, 'G': 131, 'B-': 80, 'c': 52, 'f': 48, 'c#': 20, 'e': 31, 'A': 82, 'b': 32, 'C#': 19, 'e-': 11, 'A-': 30, 'g#': 13, 'b-': 16, 'B': 11}
# basically all keys are represented!