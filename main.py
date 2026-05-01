import librosa
import numpy as np
import cv2
from moviepy import VideoClip, AudioFileClip
import os
import tkinter as tk
from tkinter import filedialog

def create_mv(audio_path, output_path):
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("Extracting audio features...")
    # Extract RMS energy
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Normalize RMS to [0, 1]
    rms_norm = rms / (np.max(rms) + 1e-6)
    
    # Extract Beats
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    
    fps = 30
    width, height = 1280, 720
    
    print("Rendering video frames...")
    def make_frame(t):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get RMS at this time
        frame_idx = int((t * sr) / hop_length)
        if frame_idx >= len(rms_norm):
            frame_idx = len(rms_norm) - 1
        
        current_rms = rms_norm[frame_idx]
        
        # Background pulse based on beat
        bg_intensity = 0
        for b in beat_times:
            if 0 <= t - b < 0.2:
                bg_intensity = int(255 * (1.0 - (t - b) / 0.2))
                break
        
        frame[:, :] = (bg_intensity // 4, bg_intensity // 5, bg_intensity // 2)
        
        # Central pulsing circle based on RMS
        center = (width // 2, height // 2)
        radius = int(50 + 300 * current_rms)
        color = (200, 100 + int(155*current_rms), 255)
        
        cv2.circle(frame, center, radius, color, -1)
        
        # Draw waveform ring
        num_points = 100
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            wave_amp = 0
            if frame_idx - 5 + i % 10 >= 0 and frame_idx - 5 + i % 10 < len(rms_norm):
                wave_amp = rms_norm[frame_idx - 5 + i % 10]
            
            r_outer = radius + 20 + 100 * wave_amp
            x = int(center[0] + r_outer * np.cos(angle))
            y = int(center[1] + r_outer * np.sin(angle))
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            
        return frame
        
    # Generate video clip
    video = VideoClip(make_frame, duration=duration)
    audio = AudioFileClip(audio_path)
    video = video.with_audio(audio)
    
    print(f"Writing final video to: {output_path}")
    video.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")
    print("Done!")

if __name__ == "__main__":
    # Hide the root tkinter window
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
        # Determine output filename based on input
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_name = f"{base_name}_mv.mp4"
        create_mv(file_path, output_name)
