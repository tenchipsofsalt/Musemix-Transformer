import functions
import settings
import numpy as np
import decode2


# Verify your training midi dataset has no errors
def verify(events):
    cur_vel = 0
    cur_tempo = 0  # change to 0 if you want no default tempo (wait for tempo token)
    cur_pause = 0
    cur_notes = []
    pedal = False
    event_count = len(events)

    for i in range(event_count):
        event = events[i]
        if event <= settings.tempo_offset:
            cur_vel = decode2.bin2vel(event)
        elif event <= settings.note_offset:
            cur_tempo = (event - settings.tempo_offset - 1) * settings.tempo_step + settings.min_tempo
            cur_pause = 0
        elif event <= settings.pause_offset:
            if cur_tempo == 0:
                print('[E] No tempo found, continuing...')
                continue
            note = event - settings.note_offset - 1
            if note <= settings.note_dim // 2:
                found_pause = False  # just checking to make sure the note isn't length 0
                for j in range(i + 1, event_count):
                    if settings.pause_offset < events[j] <= settings.pedal_offset:
                        found_pause = True
                    elif events[j] == event + settings.note_dim // 2:  # found note off
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
                note -= settings.note_dim // 2 + 1
                if note in cur_notes:
                    cur_notes.remove(note)
                    cur_pause = 0
        elif event <= settings.pedal_offset:
            if cur_tempo == 0:
                print('[E] No tempo found, continuing...')
                continue
            cur_pause += decode2.steps2ticks(event - settings.pause_offset, cur_tempo)
        else:
            value = event - settings.pedal_offset
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


if __name__ == '__main__':
    files = functions.get_files('Music/Beethoven/wordEvents', '.npy')
    for file in files:
        verify(np.load(file))
        print(f'Done with file {file}.')