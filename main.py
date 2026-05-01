import librosa
import numpy as np
import cv2
import colorsys
from moviepy import VideoClip, AudioFileClip
import os
import tkinter as tk
from tkinter import filedialog

def draw_polygon(img, center, radius, sides, color, thickness=2, rotation=0):
    points = []
    for i in range(sides):
        angle = 2 * np.pi * i / sides + rotation
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append([x, y])
    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, color, thickness, cv2.LINE_AA)
    return points

def draw_star(img, center, radius_in, radius_out, points_count, color, thickness=2, rotation=0):
    points = []
    for i in range(points_count * 2):
        radius = radius_out if i % 2 == 0 else radius_in
        angle = np.pi * i / points_count + rotation
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append([x, y])
    points = np.array(points, np.int32)
    cv2.polylines(img, [points], True, color, thickness, cv2.LINE_AA)

def create_mv(audio_path, output_path):
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("Separating components...")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    hop_length = 512
    
    # --- Feature Extraction ---
    print("Extracting features...")
    S_perc = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=128, fmax=8000, hop_length=hop_length)
    perc_norm = np.clip((librosa.power_to_db(S_perc, ref=np.max) + 80) / 80.0, 0, 1)
    perc_bass = np.mean(perc_norm[0:15, :], axis=0)
    perc_bass = (perc_bass / (np.max(perc_bass) + 1e-6)) ** 3
    perc_high = np.mean(perc_norm[60:, :], axis=0)
    perc_high = (perc_high / (np.max(perc_high) + 1e-6)) ** 2

    S_harm = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=128, fmax=8000, hop_length=hop_length)
    harm_norm = np.clip((librosa.power_to_db(S_harm, ref=np.max) + 80) / 80.0, 0, 1)
    harm_mid = np.mean(harm_norm[10:80, :], axis=0)
    harm_smooth = np.convolve(harm_mid, np.ones(5)/5, mode='same')
    
    # --- Section Detection ---
    print("Analyzing song structure...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Find peaks in novelty (section transitions)
    # Using a simple peak detection on moving average of onset strength
    novelty = np.convolve(onset_env, np.ones(100)/100, mode='same')
    peaks = librosa.util.peak_pick(novelty, pre_max=50, post_max=50, pre_avg=50, post_avg=50, delta=0.1, wait=200)
    section_markers = [0] + list(peaks) + [len(onset_env)]
    
    # --- Melody Extraction (CQT) ---
    print("Extracting melody line...")
    cqt = np.abs(librosa.cqt(y_harmonic, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12))
    cqt_norm = np.clip((librosa.amplitude_to_db(cqt, ref=np.max) + 80) / 80.0, 0, 1)
    melody_pitch = np.argmax(cqt_norm, axis=0)
    melody_energy = np.max(cqt_norm, axis=0)
    # Smooth melody pitch to avoid jitter
    melody_pitch_smooth = np.convolve(melody_pitch, np.ones(5)/5, mode='same')

    # --- Design Assets ---
    palettes = [
        [(40, 20, 10), (100, 50, 200), (255, 100, 255)], # Deep Purple/Neon
        [(10, 30, 40), (20, 150, 100), (100, 255, 200)], # Cyber Teal
        [(40, 10, 10), (200, 100, 50), (255, 200, 100)], # Sunset Gold
        [(20, 10, 40), (50, 100, 250), (150, 200, 255)], # Electric Blue
    ]
    shapes = ["circle", "hexagon", "star", "diamond", "rings"]

    fps = 30
    width, height = 1280, 720
    num_history = 80 # Number of points in ribbon

    print("Rendering video frames...")
    def make_frame(t):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        frame_idx = int((t * sr) / hop_length)
        if frame_idx >= len(perc_bass): frame_idx = len(perc_bass) - 1
        
        # Determine current section
        section_idx = 0
        for i in range(len(section_markers)-1):
            if section_markers[i] <= frame_idx < section_markers[i+1]:
                section_idx = i
                break
        
        # Pick section assets
        palette = palettes[section_idx % len(palettes)]
        shape_type = shapes[section_idx % len(shapes)]
        
        p_bass = perc_bass[frame_idx]
        p_high = perc_high[frame_idx]
        h_mid = harm_smooth[frame_idx]
        
        # 1. Background (Section Palette)
        bg_color = palette[0]
        # Pulse brightness with harmonic mid
        alpha = 0.2 + 0.8 * h_mid
        frame[:, :] = [int(c * alpha) for c in bg_color]
        
        center = (width // 2, height // 2)

        # 2. Glowing Ribbon (Melody)
        # We draw multiple layers with different thickness and alpha
        history_start = max(0, frame_idx - num_history)
        hist_p = melody_pitch_smooth[history_start:frame_idx]
        hist_e = melody_energy[history_start:frame_idx]
        
        if len(hist_p) > 1:
            # Create a glow layer
            overlay = frame.copy()
            for layer in range(3): # 3 layers of glow
                thickness = (3 - layer) * 4
                alpha_layer = 0.2 + 0.3 * (1 - layer/3.0)
                
                for i in range(1, len(hist_p)):
                    if hist_e[i] < 0.2: continue
                    x_prev = int(width * (i - 1) / num_history)
                    x_curr = int(width * i / num_history)
                    y_prev = int(height * 0.8 - (height * 0.6 * hist_p[i-1] / 84))
                    y_curr = int(height * 0.8 - (height * 0.6 * hist_p[i] / 84))
                    
                    color = palette[2] if layer == 0 else palette[1]
                    cv2.line(overlay, (x_prev, y_prev), (x_curr, y_curr), color, thickness, cv2.LINE_AA)
            
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # 3. Complex Center Object
        rot = t * 2.0
        # Dynamic complexity based on bass
        c_radius = int(80 + 150 * p_bass)
        c_color = palette[2]
        
        if shape_type == "hexagon":
            draw_polygon(frame, center, c_radius, 6, c_color, 3, rot)
            draw_polygon(frame, center, int(c_radius * 0.7), 6, palette[1], 2, -rot * 1.5)
        elif shape_type == "star":
            draw_star(frame, center, int(c_radius * 0.4), c_radius, 5, c_color, 3, rot)
            cv2.circle(frame, center, int(c_radius * 0.2), palette[1], -1)
        elif shape_type == "diamond":
            draw_polygon(frame, center, c_radius, 4, c_color, 3, rot)
            draw_polygon(frame, center, int(c_radius * 0.5), 4, palette[1], 5, -rot)
        elif shape_type == "rings":
            for r in range(3):
                cv2.circle(frame, center, int(c_radius * (1.0 - r * 0.3)), c_color if r % 2 == 0 else palette[1], 3)
                # Add small "satellites" on rings
                sat_angle = rot + r * np.pi/2
                sx = int(center[0] + int(c_radius * (1.0 - r * 0.3)) * np.cos(sat_angle))
                sy = int(center[1] + int(c_radius * (1.0 - r * 0.3)) * np.sin(sat_angle))
                cv2.circle(frame, (sx, sy), 10, palette[2], -1)
        else: # Default complex circle
            cv2.circle(frame, center, c_radius, c_color, 3)
            cv2.circle(frame, center, int(c_radius * 0.8), palette[1], 2)
            # Inner pulsing core
            cv2.circle(frame, center, int(c_radius * 0.4 * (1 + 0.2 * h_mid)), palette[2], -1)

        # Center object aura
        cv2.circle(frame, center, c_radius + 10, (255, 255, 255), max(1, int(5 * h_mid)))

        return frame
        
    video = VideoClip(make_frame, duration=duration)
    audio = AudioFileClip(audio_path)
    video = video.with_audio(audio)
    
    print(f"Writing final video to: {output_path}")
    video.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")
    print("Done!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select an audio file...")
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a"), ("All Files", "*.*")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_name = os.path.join("output", f"{base_name}_mv.mp4")
        create_mv(file_path, output_name)
