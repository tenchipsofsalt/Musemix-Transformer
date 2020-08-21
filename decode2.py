import mido
import settings
import numpy as np


# converts steps to ticks at a certain BPM
def steps2ticks(steps, bpm):
    return int(mido.second2tick(steps * 60 / (settings.sampling_freq * bpm), settings.ticks_per_beat,
                                mido.bpm2tempo(bpm)))


# velocity bin to velocity
def bin2vel(v_bin):
    if v_bin == 1:
        return 55
    elif v_bin == 2:
        return 65
    elif v_bin == 3:
        return 75
    elif v_bin == 4:
        return 85
    elif v_bin == 5:
        return 95
    else:
        return 105


# decode numerical events and output midi
def decode(events, output_name, touhou=False):
    mid = mido.MidiFile(ticks_per_beat=settings.ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # just some header stuff for fun
    if touhou:
        track.append(mido.Message('control_change', control=0, value=0, time=0))
        track.append(mido.Message('control_change', control=32, value=0, time=0))
        track.append(mido.Message('control_change', control=7, value=125, time=0))
        track.append(mido.Message('control_change', control=11, value=120, time=0))
        track.append(mido.Message('control_change', control=10, value=32, time=0))
        track.append(mido.Message('control_change', control=91, value=50, time=0))
        track.append(mido.Message('control_change', control=93, value=40, time=0))
    else:
        track.append(mido.Message('control_change', control=7, value=127, time=0))
        track.append(mido.Message('control_change', control=10, value=64, time=0))
    cur_vel = 0
    cur_tempo = 120  # change to 0 if you want no default tempo (wait for tempo token)
    cur_pause = 0
    cur_notes = []
    pedal = False
    event_count = len(events)

    for i in range(event_count):
        event = events[i]
        if event <= settings.tempo_offset:
            cur_vel = bin2vel(event)
        elif event <= settings.note_offset:
            cur_tempo = (event - settings.tempo_offset - 1) * settings.tempo_step + settings.min_tempo
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(cur_tempo), time=cur_pause))
            cur_pause = 0
        elif event <= settings.pause_offset:
            if cur_tempo == 0:
                print("No tempo found, continuing...")
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
                            track.append(mido.Message('note_on', note=note, velocity=cur_vel, time=cur_pause))
                            cur_pause = 0
                        else:
                            print(f'Found note {note} of 0 length at position {i}, not adding...')
                        break
                    elif events[j] == event:
                        print(f'Found note {note} without any note_off at position {i}, not adding...')
                        break
            else:
                note -= settings.note_dim // 2
                if note in cur_notes:
                    cur_notes.remove(note)
                    track.append(mido.Message('note_on', note=note, velocity=0, time=cur_pause))
                    cur_pause = 0
        elif event <= settings.pedal_offset:
            if cur_tempo == 0:
                print("No tempo found, continuing...")
                continue
            cur_pause += steps2ticks(event - settings.pause_offset, cur_tempo)
        else:
            value = event - settings.pedal_offset
            if value == 3:
                # piano
                track.append(mido.Message('program_change', program=0, time=cur_pause))
                cur_pause = 0
            elif value == 4:
                track.append(mido.MetaMessage('end_of_track', time=cur_pause))
                break
            elif value == 1 and not pedal:
                if cur_tempo == 0:
                    print("No tempo found, continuing...")
                    continue
                track.append(mido.Message('control_change', time=cur_pause, control=64, value=127))
                cur_pause = 0
                pedal = True
            elif value == 2 and pedal:
                if cur_tempo == 0:
                    print("No tempo found, continuing...")
                    continue
                track.append(mido.Message('control_change', time=cur_pause, control=64, value=0))
                cur_pause = 0
                pedal = False
    mid.save(f'results/{output_name}')


if __name__ == '__main__':
    elise = np.load('Music/touhou/wordEvents/sh01_01.mid0.npy')
    decode(elise, 'touhou.mid')
