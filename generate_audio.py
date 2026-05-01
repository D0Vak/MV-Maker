import numpy as np
from scipy.io import wavfile

def generate_kick(duration=0.2, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), False)
    freq = np.linspace(150, 40, len(t))
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 15)
    return signal

def generate_hihat(duration=0.1, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), False)
    signal = np.random.normal(0, 1, len(t)) * np.exp(-t * 30)
    for i in range(1, len(signal)):
        signal[i] = signal[i] - 0.9 * signal[i-1]
    return signal

sr = 44100
bpm = 120
beat_interval = 60.0 / bpm
total_beats = 20
total_duration = int(total_beats * beat_interval)
audio_track = np.zeros(int(total_duration * sr))

kick = generate_kick(sr=sr)
hihat = generate_hihat(sr=sr)

for i in range(total_beats):
    start = int(i * beat_interval * sr)
    if i % 2 == 0:
        end = start + len(kick)
        audio_track[start:end] += kick[:min(len(kick), len(audio_track)-start)]
    else:
        end = start + len(hihat)
        audio_track[start:end] += hihat[:min(len(hihat), len(audio_track)-start)]

audio_track = audio_track / np.max(np.abs(audio_track))
wavfile.write("data/sample.wav", sr, (audio_track * 32767).astype(np.int16))
print("data/sample.wav generated.")
