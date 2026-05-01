import librosa
import numpy as np
import cv2
import colorsys
from moviepy import VideoClip, AudioFileClip
import os
import tkinter as tk
from tkinter import filedialog

def create_mv(audio_path, output_path):
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("Separating Harmonic and Percussive components...")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    print("Extracting basic features...")
    hop_length = 512
    n_mels = 128
    
    # 1. Percussive Features
    S_perc = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length)
    S_perc_dB = librosa.power_to_db(S_perc, ref=np.max)
    S_perc_norm = np.clip((S_perc_dB + 80) / 80.0, 0, 1)
    
    perc_bass = np.mean(S_perc_norm[0:15, :], axis=0)
    perc_bass = perc_bass / (np.max(perc_bass) + 1e-6)
    perc_bass = perc_bass ** 3
    
    perc_high = np.mean(S_perc_norm[60:, :], axis=0)
    perc_high = perc_high / (np.max(perc_high) + 1e-6)
    perc_high = perc_high ** 2
    
    # 2. Harmonic Features
    S_harm = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length)
    S_harm_dB = librosa.power_to_db(S_harm, ref=np.max)
    S_harm_norm = np.clip((S_harm_dB + 80) / 80.0, 0, 1)
    
    harm_mid = np.mean(S_harm_norm[10:80, :], axis=0)
    harm_mid = harm_mid / (np.max(harm_mid) + 1e-6)
    
    window_size = 5
    harm_smooth = np.convolve(harm_mid, np.ones(window_size)/window_size, mode='same')
    
    # 3. Mood/Intensity (Long-term Energy and Complexity)
    print("Calculating mood (long-term intensity)...")
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    
    rms_norm = rms / (np.max(rms) + 1e-6)
    bw_norm = spec_bw / (np.max(spec_bw) + 1e-6)
    
    # Base intensity is a mix of volume and spectral complexity
    intensity = (rms_norm + bw_norm) / 2.0
    
    # Smooth over ~2 seconds to get slow mood changes
    smooth_frames = int(2.0 * sr / hop_length)
    if smooth_frames == 0: smooth_frames = 1
    long_intensity = np.convolve(intensity, np.ones(smooth_frames)/smooth_frames, mode='same')
    long_intensity = long_intensity / (np.max(long_intensity) + 1e-6)
    
    # 4. Melody Extraction (CQT for piano-roll mapping)
    print("Extracting melody (CQT)...")
    cqt = np.abs(librosa.cqt(y_harmonic, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt_norm = np.clip((cqt_db + 80) / 80.0, 0, 1)
    
    melody_pitch = np.argmax(cqt_norm, axis=0)
    melody_energy = np.max(cqt_norm, axis=0)
    
    fps = 30
    width, height = 1280, 720
    
    # Number of historical points to draw for the melody line
    num_history = 100 
    
    print("Rendering video frames...")
    def make_frame(t):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        frame_idx = int((t * sr) / hop_length)
        max_idx = len(perc_bass) - 1
        if frame_idx >= max_idx:
            frame_idx = max_idx
            
        p_bass = perc_bass[frame_idx]
        p_high = perc_high[frame_idx]
        h_mid = harm_smooth[frame_idx]
        mood = long_intensity[frame_idx] # Current mood (0.0 ~ 1.0)
        
        # --- Dynamic Color Palette (HSV based on Mood) ---
        # Quiet: Fixed Hue (Blue), high Saturation, low Value
        # Chorus/Loud: Rotating Hue, high Saturation, max Value
        base_hue = (t * 0.1) % 1.0 if mood > 0.5 else 0.65 
        saturation = 0.7 + 0.3 * mood
        value = 0.2 + 0.8 * mood
        
        # Background
        bg_r, bg_g, bg_b = colorsys.hsv_to_rgb(base_hue, saturation, value * h_mid * 0.5)
        frame[:, :] = (int(bg_b*255), int(bg_g*255), int(bg_r*255))
        
        center = (width // 2, height // 2)
        
        # --- Melody Line Drawing (Piano-Roll style) ---
        history_start = max(0, frame_idx - num_history)
        history_pitch = melody_pitch[history_start:frame_idx]
        history_energy = melody_energy[history_start:frame_idx]
        
        if len(history_pitch) > 1:
            for i in range(1, len(history_pitch)):
                p_val_prev = history_pitch[i-1]
                p_val_curr = history_pitch[i]
                e_val_curr = history_energy[i]
                
                # Skip drawing if the energy is too low (rest)
                if e_val_curr < 0.25: continue
                
                # Flow from left to right
                x_prev = int(width * (i - 1) / num_history)
                x_curr = int(width * i / num_history)
                
                # Map pitch (0-84) to Y axis (leave top/bottom margins)
                y_prev = int(height * 0.8 - (height * 0.6 * p_val_prev / 84))
                y_curr = int(height * 0.8 - (height * 0.6 * p_val_curr / 84))
                
                line_hue = (base_hue + 0.1) % 1.0
                lr, lg, lb = colorsys.hsv_to_rgb(line_hue, 0.9, e_val_curr)
                line_color = (int(lb*255), int(lg*255), int(lr*255))
                
                line_thickness = max(1, int(3 * e_val_curr * (1 + mood)))
                
                cv2.line(frame, (x_prev, y_prev), (x_curr, y_curr), line_color, line_thickness)
                # Draw node
                cv2.circle(frame, (x_curr, y_curr), line_thickness + 2, (255, 255, 255), -1)

        # --- Dynamic Scale factor for intensity ---
        dynamic_scale = 0.5 + 1.5 * mood
        
        # --- Surround Particles ---
        num_particles = 150
        base_radius = 100 + 100 * h_mid + 50 * p_bass
        
        for i in range(num_particles):
            angle = 2 * np.pi * i / num_particles
            idx_offset = (frame_idx - 10 + i % 20) % (max_idx + 1)
            local_p_high = perc_high[idx_offset]
            
            p_radius = base_radius + 150 * local_p_high * dynamic_scale * (1 if i % 2 == 0 else -0.5)
            x = int(center[0] + p_radius * np.cos(angle))
            y = int(center[1] + p_radius * np.sin(angle))
            
            p_hue = (base_hue + 0.5) % 1.0 # Complementary color
            pr, pg, pb = colorsys.hsv_to_rgb(p_hue, 1.0, local_p_high)
            color = (int(pb*255), int(pg*255), int(pr*255))
            
            size = int(2 + 5 * local_p_high * dynamic_scale)
            cv2.circle(frame, (x, y), size, color, -1)
            
            if i > 0:
                prev_angle = 2 * np.pi * (i-1) / num_particles
                prev_idx_offset = (frame_idx - 10 + (i-1) % 20) % (max_idx + 1)
                prev_p_high = perc_high[prev_idx_offset]
                prev_p_radius = base_radius + 150 * prev_p_high * dynamic_scale * (1 if (i-1) % 2 == 0 else -0.5)
                px = int(center[0] + prev_p_radius * np.cos(prev_angle))
                py = int(center[1] + prev_p_radius * np.sin(prev_angle))
                cv2.line(frame, (px, py), (x, y), (255, 255, 255), 1)
                
        # Close particle loop
        first_angle = 0
        first_idx_offset = (frame_idx - 10) % (max_idx + 1)
        first_p_high = perc_high[first_idx_offset]
        first_p_radius = base_radius + 150 * first_p_high * dynamic_scale * 1
        fx = int(center[0] + first_p_radius * np.cos(first_angle))
        fy = int(center[1] + first_p_radius * np.sin(first_angle))
        
        last_angle = 2 * np.pi * (num_particles-1) / num_particles
        last_idx_offset = (frame_idx - 10 + (num_particles-1) % 20) % (max_idx + 1)
        last_p_high = perc_high[last_idx_offset]
        last_p_radius = base_radius + 150 * last_p_high * dynamic_scale * (1 if (num_particles-1) % 2 == 0 else -0.5)
        lx = int(center[0] + last_p_radius * np.cos(last_angle))
        ly = int(center[1] + last_p_radius * np.sin(last_angle))
        cv2.line(frame, (lx, ly), (fx, fy), (255, 255, 255), 1)
        
        # --- Main Circle ---
        main_radius = int(50 + 150 * p_bass * dynamic_scale)
        mr, mg, mb = colorsys.hsv_to_rgb(base_hue, saturation, 0.5 + 0.5 * p_bass)
        main_color = (int(mb*255), int(mg*255), int(mr*255))
        
        cv2.circle(frame, center, main_radius, main_color, -1)
        edge_thickness = max(2, int(2 + 10 * h_mid * dynamic_scale))
        cv2.circle(frame, center, main_radius, (255, 255, 255), edge_thickness)
        
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
    
    print("Please select an audio file (mp3, wav, m4a)...")
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
