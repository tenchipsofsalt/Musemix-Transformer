import mido
import numpy as np
import settings
import functions
from os import path, makedirs


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
    for i in range(len(events)):
        event = events[i]
        if max(settings.note_offset, settings.note_offset - shift) < event <= \
                min(settings.pause_offset, settings.pause_offset - shift) - settings.note_dim // 2 or \
                max(settings.note_offset, settings.note_offset - shift) + settings.note_dim // 2 < event <= \
                min(settings.pause_offset, settings.pause_offset - shift):
            events[i] += shift
    return events


def process(file_path):  # needs a file path with / instead of \\.
    filename = file_path.split('/')[-1]
    if path.exists(file_path.rsplit('/', 1)[0] + f'/wordEvents/{filename}.npy'):
        print('File already exists, skipping.')
        return

    try:
        midi = mido.MidiFile(file_path)
    except Exception as error:
        print(error)
        return

    # remove unknown meta messages
    for track in midi.tracks:
        to_remove = []
        for i in range(len(track)):
            if track[i].type == 'unknown_meta':
                to_remove.append(track[i])
        for removed in to_remove:
            track.remove(removed)

    cur_tempo = 0
    cur_vel = -1
    pause_time = 0
    prev_tempo = 0
    pedal = False
    events = [settings.pedal_offset + 3]
    accepted_ch = []
    cur_notes = []
    queue_notes = []  # keep notes of current timestep in memory to make sure notes are not of 0 length

    for message in midi:
        if message.type == 'program_change':
            if message.program < 8:  # is in piano family
                accepted_ch.append(message.channel)
        # tempo related

        pause_time += message.time
        if message.type == 'set_tempo':
            if cur_tempo == 0:  # find first tempo
                cur_tempo = clip_tempo(int(np.round(mido.tempo2bpm(message.tempo))))  # set tempo to first tempo
                events.append(settings.tempo_offset + tempo2bin(cur_tempo))  # append tempo event
            else:  # wait for a non-tempo event to determine how much time has passed and set new tempo
                # deal with time at the end of a long set of tempo changes
                prev_tempo = clip_tempo(int(np.round(mido.tempo2bpm(message.tempo))))  # keep last tempo in memory
            continue

        if pause_time != 0:  # if this is true, we need to add a pause token
            if message.type != 'control_change' or (message.type == 'control_change' and message.control == 64):
                queue_notes = []
                events.append(settings.pause_offset + time2steps(pause_time, cur_tempo))  # we can assume this is right
                # because cc length is always 0, so if it's not a tempo shift it MUST have some pause.
                pause_time = 0

        if prev_tempo != 0:  # if previous event was a tempo shift and this was NOT a tempo shift
            if tempo2bin(prev_tempo) != tempo2bin(cur_tempo):
                events.append(settings.tempo_offset + tempo2bin(prev_tempo))
            cur_tempo = prev_tempo
            prev_tempo = 0

        try:
            if message.channel not in accepted_ch:
                continue
        except AttributeError:
            pass

        if message.type == 'note_on' and message.velocity != 0:  # makes sure this is a note on event,
            # since it's AND it will exit if note is not note_on so this shouldn't throw errors.
            velocity = velocity2bin(message.velocity)
            if velocity != cur_vel:  # if velocity has changed, make new event
                cur_vel = velocity
                events.append(settings.velocity_offset + cur_vel)
            value = settings.note_offset + message.note + 1
            if value not in cur_notes:
                cur_notes.append(value)
                queue_notes.append(value)
                events.append(value)  # append note on
            else:
                # print(f'Note {message.note + 1} was not appended because it is already playing.')
                pass

        elif message.type == 'note_off' or message.type == 'note_on':  # note_on with v = 0 is same as note_off
            value = settings.note_offset + message.note + 1
            if value in queue_notes:
                queue_notes.remove(value)
                events.reverse()
                events.remove(value)
                events.reverse()
                # print(f'Removed note {message.note + 1} because it had 0 length.')
                cur_notes.remove(value)
                continue
            try:
                cur_notes.remove(value)
                events.append(value + settings.note_dim // 2)  # append note off
            except ValueError:
                # print(f"Note {message.note + 1} was not appended because there was no onset.")
                pass

        elif message.type == 'control_change' and message.control == 64:  # what control changes do we need to note?
            if message.value > 63 and not pedal:
                events.append(settings.pedal_offset + 1)
                pedal = True
            elif message.value <= 63 and pedal:
                events.append(settings.pedal_offset + 2)
                pedal = False
    # stack waits together, cap at ceiling if necessary
    fixed_events = []
    i = 0
    while i < len(events):
        if settings.pause_offset < events[i] <= settings.pedal_offset:
            wait_event = events[i]
            i += 1
            while i < len(events) and settings.pause_offset < events[i] <= settings.pedal_offset:
                # print(f'Stacking pauses... length is now {wait_event - settings.pause_offset}')
                wait_event = min(settings.pedal_offset, wait_event + events[i] - settings.pause_offset)
                i += 1
            fixed_events.append(wait_event)
        else:
            fixed_events.append(events[i])
            i += 1
    fixed_events.append(settings.pedal_offset + 4)
    if len(fixed_events) < settings.seq_len:
        print(f'File {filename} is too short, skipping.')
        return
    # for shift in settings.shifts:
    #     temp_events = note_shift(fixed_events.copy(), shift)
    np.save(file_path.rsplit('/', 1)[0] + f'/wordEvents/{filename}', fixed_events)


if __name__ == '__main__':
    dirs = settings.dataset_dir
    for directory in dirs:
        files = functions.get_files(f'Music/{directory}/', '.midi')
        count = 0
        if not path.exists(f'Music/{directory}/wordEvents/'):
            makedirs(f'Music/{directory}/wordEvents/')
        for file in files:
            count += 1
            process(file)
            print(f'Done with file {file} Progress: {count} / {len(files)}.')

        if len(functions.get_files(f'Music/{directory}/wordEvents/', '.npy')) == 0:
            print(f'[E] Directory {directory} has no valid wordEvents')
