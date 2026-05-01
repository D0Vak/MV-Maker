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

def draw_ribbon(frame, hist_p, hist_e, color, num_history, mode="LR", layers=3, alpha_mult=1.0):
    width, height = frame.shape[1], frame.shape[0]
    overlay = frame.copy()
    
    for layer in range(layers):
        thickness = (layers - layer) * 5
        alpha_layer = (0.2 + 0.3 * (1 - layer/layers)) * alpha_mult
        
        for i in range(1, len(hist_p)):
            if hist_e[i] < 0.2: continue
            
            # Mapping based on mode
            if mode == "LR":
                x_prev = int(width * (i - 1) / num_history)
                x_curr = int(width * i / num_history)
                y_prev = int(height * (0.1 + 0.8 * (1 - hist_p[i-1] / 84)))
                y_curr = int(height * (0.1 + 0.8 * (1 - hist_p[i] / 84)))
            elif mode == "TB": # Top to Bottom (Left side)
                y_prev = int(height * (i - 1) / num_history)
                y_curr = int(height * i / num_history)
                x_prev = int(width * (0.05 + 0.25 * (hist_p[i-1] / 84)))
                x_curr = int(width * (0.05 + 0.25 * (hist_p[i] / 84)))
            elif mode == "BT": # Bottom to Top (Right side)
                y_prev = int(height * (1 - (i - 1) / num_history))
                y_curr = int(height * (1 - i / num_history))
                x_prev = int(width * (0.7 + 0.25 * (hist_p[i-1] / 84)))
                x_curr = int(width * (0.7 + 0.25 * (hist_p[i] / 84)))
            else: # Default LR
                x_prev, x_curr, y_prev, y_curr = 0, 0, 0, 0

            cv2.line(overlay, (x_prev, y_prev), (x_curr, y_curr), color, thickness, cv2.LINE_AA)
            
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

def create_mv(audio_path, output_path):
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("Separating components...")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    hop_length = 512
    
    # --- Feature Extraction ---
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
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    novelty = np.convolve(onset_env, np.ones(100)/100, mode='same')
    peaks = librosa.util.peak_pick(novelty, pre_max=50, post_max=50, pre_avg=50, post_avg=50, delta=0.1, wait=200)
    section_markers = [0] + list(peaks) + [len(onset_env)]
    
    # --- Melody Extraction (Main & Sub) ---
    print("Extracting melody lines...")
    cqt = np.abs(librosa.cqt(y_harmonic, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12))
    cqt_norm = np.clip((librosa.amplitude_to_db(cqt, ref=np.max) + 80) / 80.0, 0, 1)
    
    melody_pitch = np.argmax(cqt_norm, axis=0)
    melody_energy = np.max(cqt_norm, axis=0)
    
    # Mask strongest pitch to find second strongest
    cqt_masked = cqt_norm.copy()
    for i in range(len(melody_pitch)):
        cqt_masked[melody_pitch[i], i] = 0
    sub_melody_pitch = np.argmax(cqt_masked, axis=0)
    sub_melody_energy = np.max(cqt_masked, axis=0)
    
    # Smoothing
    m_p_smooth = np.convolve(melody_pitch, np.ones(5)/5, mode='same')
    s_p_smooth = np.convolve(sub_melody_pitch, np.ones(5)/5, mode='same')

    # --- Assets ---
    palettes = [
        [(40, 20, 10), (100, 50, 200), (255, 100, 255)], # Purple
        [(10, 30, 40), (20, 150, 100), (100, 255, 200)], # Teal
        [(40, 10, 10), (200, 100, 50), (255, 200, 100)], # Sunset
        [(20, 10, 40), (50, 100, 250), (150, 200, 255)], # Blue
    ]
    shapes = ["circle", "hexagon", "star", "diamond", "rings"]
    ribbon_modes = ["TB", "BT", "LR"] # Order per section

    fps = 30
    width, height = 1280, 720
    num_history = 80

    print("Rendering video frames...")
    def make_frame(t):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        frame_idx = int((t * sr) / hop_length)
        if frame_idx >= len(perc_bass): frame_idx = len(perc_bass) - 1
        
        section_idx = 0
        for i in range(len(section_markers)-1):
            if section_markers[i] <= frame_idx < section_markers[i+1]:
                section_idx = i
                break
        
        palette = palettes[section_idx % len(palettes)]
        shape_type = shapes[section_idx % len(shapes)]
        mode = ribbon_modes[section_idx % len(ribbon_modes)]
        
        p_bass = perc_bass[frame_idx]
        p_high = perc_high[frame_idx]
        h_mid = harm_smooth[frame_idx]
        
        # 1. Background
        bg_color = palette[0]
        alpha = 0.2 + 0.8 * h_mid
        frame[:, :] = [int(c * alpha) for c in bg_color]
        
        center = (width // 2, height // 2)

        # 2. Sub-Melody Edge Visuals
        # Pulse bars at the edges based on sub-melody energy
        s_energy = sub_melody_energy[frame_idx]
        bar_width = int(20 + 80 * s_energy)
        cv2.rectangle(frame, (0, 0), (bar_width, height), palette[1], -1)
        cv2.rectangle(frame, (width - bar_width, 0), (width, height), palette[1], -1)
        # Glow for edges
        edge_overlay = frame.copy()
        cv2.rectangle(edge_overlay, (0, 0), (bar_width + 20, height), palette[2], -1)
        cv2.rectangle(edge_overlay, (width - bar_width - 20, 0), (width, height), palette[2], -1)
        cv2.addWeighted(edge_overlay, 0.3 * s_energy, frame, 1.0, 0, dst=frame)

        # 3. Dynamic Ribbons
        history_start = max(0, frame_idx - num_history)
        hist_p = m_p_smooth[history_start:frame_idx]
        hist_e = melody_energy[history_start:frame_idx]
        hist_sp = s_p_smooth[history_start:frame_idx]
        hist_se = sub_melody_energy[history_start:frame_idx]
        
        if len(hist_p) > 1:
            # Complexity & Count per section
            # LR mode (Chorus) = Multiple ribbons, High complexity
            if mode == "LR":
                # Main
                draw_ribbon(frame, hist_p, hist_e, palette[2], num_history, "LR", layers=5)
                # Sub
                draw_ribbon(frame, hist_sp, hist_se, palette[1], num_history, "LR", layers=3, alpha_mult=0.6)
            else:
                # Verse (TB or BT) = Single ribbon, Moderate complexity
                draw_ribbon(frame, hist_p, hist_e, palette[2], num_history, mode, layers=3)
                if s_energy > 0.5: # Only show sub if strong in Verse
                    draw_ribbon(frame, hist_sp, hist_se, palette[1], num_history, mode, layers=2, alpha_mult=0.4)

        # 4. Complex Center Object
        rot = t * 2.0
        c_radius = int(80 + 150 * p_bass)
        c_color = palette[2]
        
        if shape_type == "hexagon":
            draw_polygon(frame, center, c_radius, 6, c_color, 3, rot)
            draw_polygon(frame, center, int(c_radius * 0.7), 6, palette[1], 2, -rot * 1.5)
        elif shape_type == "star":
            draw_star(frame, center, int(c_radius * 0.4), c_radius, 5, c_color, 3, rot)
        elif shape_type == "diamond":
            draw_polygon(frame, center, c_radius, 4, c_color, 3, rot)
            draw_polygon(frame, center, int(c_radius * 0.5), 4, palette[1], 5, -rot)
        elif shape_type == "rings":
            for r in range(3):
                cv2.circle(frame, center, int(c_radius * (1.0 - r * 0.3)), c_color if r % 2 == 0 else palette[1], 3)
        else: # Circle
            cv2.circle(frame, center, c_radius, c_color, 3)
            cv2.circle(frame, center, int(c_radius * 0.4 * (1 + 0.2 * h_mid)), palette[2], -1)

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
