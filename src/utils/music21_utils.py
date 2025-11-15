"""
Парсинг MIDI → ноты, разделение партий, создание Score.
"""
import os

import numpy as np

from music21 import converter, instrument, note, chord, stream


def midi_to_notes(file_path):
    midi = converter.parse(file_path)
    notes = []
    prev_offset = 0.0
    for n in midi.flat.notes:
        if isinstance(n, note.Note):
            pitch = n.pitch.midi
        elif isinstance(n, chord.Chord):
            pitch = n.pitches[0].midi
        else:
            continue
        step = float(n.offset - prev_offset)
        dur = float(n.quarterLength)
        notes.append({'pitch': pitch, 'step': step, 'duration': dur})
        prev_offset = n.offset
    return notes


def split_parts(notes, melody_threshold=60, chord_threshold=48):
    """
       Простейшее разделение по высоте:
       - >melody_threshold → мелодия
       - между chord_threshold и melody_threshold → аккорды
       - <chord_threshold → бас
    """
    melody, chords, bass = [], [], []
    for n in notes:
        p = n['pitch']
        if p > melody_threshold:
            melody.append(n)
        elif p < chord_threshold:
            bass.append(n)
        else:
            chords.append(n)
    return melody, chords, bass


def combine_to_score(parts):
    s = stream.Score()
    for p in parts:
        s.insert(0, p)
    return s


def get_midis_by_compositor(compositor):
    midis = []
    filepath = f"../dataset/{compositor}/"
    for i in os.listdir(filepath):
        if i.endswith(".mid"):
            tr = filepath + i
            # чтение и парсинг midi-файлов в Stream-объект библиотеки music21
            midi = converter.parse(tr)
            midis.append(midi)
    return midis


def sequence_to_part(seq, instr_name="Piano"):
    """
    Преобразует массив [[pitch, step, duration], ...] в партию music21.
    """
    part = stream.Part()
    part.insert(0, instrument.fromString(instr_name))
    offset = 0.0
    for p, s, d in seq:
        n = note.Note(int(np.clip(p, 21, 108)))# диапазон MIDI клавиатуры
        n.quarterLength = max(0.25, float(d))# минимальная длительность = 1/
        part.insert(offset, n)
        offset += float(s)
    return part
