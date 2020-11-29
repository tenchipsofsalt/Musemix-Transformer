import functions_keys2
import settings_keys2
import numpy as np
import decode_keys2


# Verify your training midi dataset has no errors
def verify(events):
    cur_vel = 0
    cur_tempo = 0  # change to 0 if you want no default tempo (wait for tempo token)
    cur_pause = 0
    cur_notes = []
    pedal = False
    event_count = len(events)
    pause = False
    for i in range(event_count):
        event = events[i]
        if event <= settings_keys2.tempo_offset:
            cur_vel = decode_keys2.bin2vel(event)
            pause = False
        elif event <= settings_keys2.note_offset:
            cur_tempo = (event - settings_keys2.tempo_offset - 1) * settings_keys2.tempo_step + settings_keys2.min_tempo
            cur_pause = 0
            pause = False
        elif event <= settings_keys2.pause_offset:
            if cur_tempo == 0:
                print('[E] No tempo found, continuing...')
                continue
            note = event - settings_keys2.note_offset - 1
            if note <= settings_keys2.note_dim // 2:
                found_pause = False  # just checking to make sure the note isn't length 0
                for j in range(i + 1, event_count):
                    if settings_keys2.pause_offset < events[j] <= settings_keys2.pedal_offset:
                        found_pause = True
                    elif events[j] == event + settings_keys2.note_dim // 2:  # found note off
                        if found_pause:
                            cur_notes.append(note)
                            cur_pause = 0
                        else:
                            print(f'[E] Found note {note} of 0 length at position {i}, not adding...')
                        break
                    elif events[j] == event:
                        print(f'[E] Found note {note} without any note_off at position {i}, not adding...')
                        break
            else:
                note -= settings_keys2.note_dim // 2 + 1
                if note in cur_notes:
                    cur_notes.remove(note)
                    cur_pause = 0
            pause = False
        elif event <= settings_keys2.pedal_offset:
            if cur_tempo == 0:
                print('[E] No tempo found, continuing...')
                continue
            if pause:
                print("consecutive pauses...")
            pause = True
            cur_pause += decode_keys2.steps2ticks(event - settings_keys2.pause_offset, cur_tempo)
        else:
            value = event - settings_keys2.pedal_offset
            if value == 3:
                cur_pause = 0
            elif value == 4:
                cur_pause = 0
                break
            elif value == 1 and not pedal:
                if cur_tempo == 0:
                    print('[E] No tempo found, continuing...')
                    continue
                cur_pause = 0
                pedal = True
            elif value == 2 and pedal:
                if cur_tempo == 0:
                    print('[E] No tempo found, continuing...')
                    continue
                cur_pause = 0
                pedal = False
            pause = False

if __name__ == '__main__':
    dirs = settings_keys2.dataset_dir
    for directory in dirs:
        files = functions_keys2.get_files(f'Music/{directory}/keyedEvents2', '.npy')
        count = 0
        for file in files:
            count += 1
            verify(np.load(file))
            print(f'Done with file {file} Progress: {count} / {len(files)}.')

        if len(files) == 0:
            print(f'[E] Directory {directory} has no valid keyedEvents2')