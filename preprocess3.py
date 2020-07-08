import mido
import numpy as np
import os
import settings
import functions


def time2steps(time, bpm):
    return min(max(int(np.round(time * bpm * settings.sampling_freq / 60)), 1), settings.pause_dim)
    # make sure pause is at least 1 and not too long


def clip_tempo(tempo):
    while tempo < settings.min_tempo:
        tempo = tempo * 2
    while tempo > settings.max_tempo:
        tempo = tempo // 2
    return tempo


def velocity2bin(velocity):
    # values from: http://nickleusmusic.blogspot.com/2013/07/midi-dynamics-note-velocity-values-for.html
    if velocity < 32:
        return 1
    elif velocity < 48:
        return 2
    elif velocity < 64:
        return 3
    elif velocity < 80:
        return 4
    elif velocity < 96:
        return 5
    else:
        return 6


def tempo2bin(tempo):
    return 1 + (tempo - settings.min_tempo) // settings.tempo_step


def note_shift(events, shift):
    for event in events:
        if max(settings.note_offset, settings.note_offset + shift) < event <= \
                min(settings.pause_offset, settings.pause_offset + shift):
            event += shift
    return events


def process(artist, file_path):  # needs a file path with / instead of \\.
    filename = file_path.split('/')[-1]
    midi = mido.MidiFile(file_path)
    cur_tempo = 0
    cur_vel = -1
    pause_time = 0
    prev_tempo = 0
    pedal = False
    events = [settings.pedal_offset + 3]
    # for pause/delay, vx for velocity
    for message in midi:
        # tempo related
        if message.type == 'set_tempo':
            if cur_tempo == 0:  # find first tempo
                cur_tempo = clip_tempo(int(np.round(mido.tempo2bpm(message.tempo))))  # set tempo to first tempo
                events.append(settings.tempo_offset + tempo2bin(cur_tempo))  # append tempo event
            else:  # wait for a non-tempo event to determine how much time has passed and set new tempo
                pause_time += message.time  # deal with time at the end of a long set of tempo changes
                prev_tempo = clip_tempo(int(np.round(mido.tempo2bpm(message.tempo))))  # keep last tempo in memory
            continue

        if message.time != 0:  # if this is true, we need to add a pause token
            pause_time += message.time
            if message.type != 'control_change' or (message.type == 'control_change' and message.control == 64):
                events.append(settings.pause_offset + time2steps(pause_time, cur_tempo))  # we can assume this is right
                # because cc length is always 0, so if it's not a tempo shift it MUST have some pause.
                pause_time = 0

        if prev_tempo != 0:  # if previous event was a tempo shift and this was NOT a tempo shift
            if tempo2bin(prev_tempo) != tempo2bin(cur_tempo):
                events.append(settings.tempo_offset + tempo2bin(prev_tempo))

            cur_tempo = prev_tempo
            prev_tempo = 0

        if message.type == 'note_on' and message.velocity != 0:  # makes sure this is a note on event,
            # since it's AND it will exit if note is not note_on so this shouldn't throw errors.
            velocity = velocity2bin(message.velocity)
            if velocity != cur_vel:  # if velocity has changed, make new event
                cur_vel = velocity
                events.append(settings.velocity_offset + cur_vel)
            events.append(settings.note_offset + message.note + 1)  # append note on

        elif message.type == 'note_off' or message.type == 'note_on':  # note_on with v = 0 is same as note_off
            events.append(settings.note_offset + message.note + 1 + settings.note_dim // 2)  # append note off

        elif message.type == 'control_change' and message.control == 64:  # what control changes do we need to note?
            if message.value > 63 and not pedal:
                events.append(settings.pedal_offset + 1)
                pedal = True
            elif message.value <= 63 and pedal:
                events.append(settings.pedal_offset + 2)
                pedal = False

    events.append(settings.pedal_offset + 4)
    for shift in settings.shifts:
        temp_events = note_shift(events.copy(), shift)
        print(max(events))
        np.save(file_path.rsplit('/', 1)[0] + f'/wordEvents/{filename}{shift}', events)


# get all files in directory
files = functions.get_files('Music/Beethoven/', '.mid')
for file in files:
    try:
        process('beethoven', file)
    except Exception as e:
        print(e)
    print(f'Done with file {file}.')
