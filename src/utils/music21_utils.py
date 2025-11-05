"""
Парсинг MIDI → ноты, разделение партий, создание Score.
"""
import os

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
