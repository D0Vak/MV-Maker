import librosa
import numpy as np
from moviepy import AudioFileClip

class AudioAnalyzer:
    @staticmethod
    def analyze(audio_path, hop=512):
        print("Starting Evolutionary Analysis...")
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except:
            print("Librosa load failed, falling back to MoviePy decoder...")
            audio_tmp = AudioFileClip(audio_path)
            sr = 44100
            y = audio_tmp.to_soundarray(fps=sr)
            if y.ndim > 1: y = y.mean(axis=1)
            audio_tmp.close()

        duration = librosa.get_duration(y=y, sr=sr)
        
        # 1. Genre & Mood Features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        tempo = float(np.mean(tempo))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y, hop_length=hop))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=hop))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 2. HPSS & Detailed Features
        y_h, y_p = librosa.effects.hpss(y)
        pb = np.mean(np.clip((librosa.power_to_db(librosa.feature.melspectrogram(y=y_p, sr=sr, n_mels=128, hop_length=hop), ref=np.max)+80)/80,0,1)[0:20,:], axis=0)
        pb = (pb / (np.max(pb)+1e-6))**2
        hm = np.convolve(np.mean(np.clip((librosa.power_to_db(librosa.feature.melspectrogram(y=y_h, sr=sr, n_mels=128, hop_length=hop), ref=np.max)+80)/80,0,1)[10:80,:], axis=0), np.ones(10)/10, mode='same')
        cqt_n = np.clip((librosa.amplitude_to_db(np.abs(librosa.cqt(y_h, sr=sr, hop_length=hop, n_bins=84)), ref=np.max)+80)/80,0,1)
        mel_p, mel_e = np.argmax(cqt_n, axis=0), np.max(cqt_n, axis=0)
        sub_e = np.max(cqt_n.copy(), axis=0)

        # 3. Structural Markers
        markers = [0]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop)
        bounds = librosa.segment.agglomerative(S, k=min(14, int(duration/8)))
        if len(beats) > 0:
            for bf in bounds:
                q_idx = round(np.abs(beats - bf).argmin() / 16) * 16
                if q_idx < len(beats): markers.append(beats[q_idx])
        markers.append(int((duration * sr) / hop))
        markers = sorted(list(set(markers)))

        # 4. Energy-based Phases
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        sec_energies = [np.mean(rms[markers[i]:markers[i+1]]) for i in range(len(markers)-1)]
        sorted_e = sorted(sec_energies)
        p1_t, p2_t = sorted_e[len(sorted_e)//3], sorted_e[2*len(sorted_e)//3]
        section_phases = [(1 if e <= p1_t else 2 if e <= p2_t else 3) for e in sec_energies]

        return {
            "y": y, "sr": sr, "duration": duration, "tempo": tempo, "beats": beats,
            "pb": pb, "hm": hm, "mel_p": mel_p, "mel_e": mel_e, "sub_e": sub_e,
            "markers": markers, "section_phases": section_phases,
            "zcr": zcr, "flatness": flatness, "rolloff": rolloff, "centroid": centroid
        }
