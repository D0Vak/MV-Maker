import librosa
import numpy as np
import cv2
from moviepy import VideoClip, AudioFileClip
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# Modular Imports
from design_system import DesignTheme, Camera, PostProcessor
from style_engines import *
from audio_analyzer import AudioAnalyzer
from video_compositor import VideoCompositor

class MVMakerEvolution:
    def __init__(self, audio_path, out_path, style_name="hybrid", theme_name="auto", ratio="16:9", timeline_script="", category="all"):
        self.audio_path, self.out_path, self.style_name, self.theme_name, self.emotion_name, self.ratio, self.timeline_script, self.category_name = \
            audio_path, out_path, style_name, theme_name, "auto", ratio, timeline_script, category
        self.pals = {
            "bright": [[(30, 20, 70), (130, 90, 240), (255, 190, 255)], [(15, 45, 55), (40, 200, 140), (160, 255, 240)]],
            "deep": [[(15, 10, 40), (70, 40, 140), (160, 90, 255)], [(30, 15, 20), (160, 70, 40), (255, 130, 90)]]
        }
        self.themes = {
            "vivid": DesignTheme("Vivid Pop", saturation=1.5, contrast=1.2, line_weight=3, glow_strength=0.8),
            "noir": DesignTheme("Noir Minimal", palette_type="deep", saturation=0.2, contrast=1.4, line_weight=1, glow_strength=0.3),
            "retro": DesignTheme("Analog Retro", noise_level=15, scanlines=1, chrom_abb=0.8, saturation=0.8),
            "dreamy": DesignTheme("Ethereal Dream", glow_strength=1.5, saturation=1.1, contrast=0.9, line_weight=2),
            "cyber": DesignTheme("Cyber Aggressive", line_weight=4, glow_strength=1.0, noise_level=10, chrom_abb=1.2, contrast=1.5),
            "ink": DesignTheme("Ink Wash", palette_type="bright", saturation=0.5, contrast=0.8, line_weight=1),
            "grunge": DesignTheme("Grunge", palette_type="deep", noise_level=30, chrom_abb=1.5, contrast=1.8),
            "pop": DesignTheme("Dynamic Pop", palette_type="bright", saturation=2.0, contrast=1.3, line_weight=5)
        }
        self.engines = {
            "fusion": StyleCyberFusion(), "anime": StyleAnimePlus(), "liquid": StyleLiquidFlow(),
            "geo": StyleGeometricChaos(), "glitch": StyleGlitchPulse(), "celestial": StyleCelestialOrbit(),
            "stormy": StyleStormyLandscape(), "slowmo": StyleRainSlowMo(), "city": StyleCyberCity(),
            "graphic": StyleGraphicArt(), "ink": StyleInkWash(), "grunge": StyleGrungeGrind(), "pop": StylePopDynamic(),
            "story": StyleStorySilhouette(), "melancholy": StyleStormyLandscape(),
            "manga": StyleMangaLayout(), "action": StyleActionSilhouette()
        }
        self.categories = {
            "all": list(self.engines.keys()),
            "geometric": ["fusion", "geo", "glitch", "graphic", "pop", "liquid"],
            "anime": ["anime", "story", "city", "stormy", "slowmo", "celestial", "manga", "action"],
            "artistic": ["ink", "grunge", "liquid"]
        }
        self.phase_styles = {
            "geometric": { 1: ["liquid", "fusion"], 2: ["fusion", "geo", "graphic"], 3: ["glitch", "pop", "geo"] },
            "anime": { 1: ["slowmo", "story", "ink", "celestial"], 2: ["anime", "city", "stormy"], 3: ["action", "manga", "grunge"] },
            "artistic": { 1: ["ink", "liquid"], 2: ["ink", "grunge"], 3: ["grunge", "glitch"] },
            "all": { 1: ["slowmo", "ink", "liquid", "celestial", "story"], 2: ["anime", "fusion", "city", "geo", "graphic"], 3: ["action", "manga", "glitch", "pop", "grunge"] }
        }
        self.emotions = {
            "joy": ["pop", "anime", "graphic"], "sadness": ["melancholy", "ink", "slowmo"],
            "anger": ["grunge", "glitch", "geo"], "peace": ["celestial", "story", "liquid"]
        }
        self.camera = Camera()

    def create(self):
        # 1. Audio Analysis
        data = AudioAnalyzer.analyze(self.audio_path)
        
        # 2. Theme Selection
        detected_theme = "vivid"
        if data["zcr"] > 0.08 or data["flatness"] > 0.04: detected_theme = "grunge"
        elif data["rolloff"] < 3000 and data["flatness"] < 0.01: detected_theme = "ink"
        elif data["tempo"] > 120 and data["flatness"] > 0.02: detected_theme = "pop"
        elif data["tempo"] < 90: detected_theme = "dreamy"
        
        final_theme_key = self.theme_name if self.theme_name != "auto" else detected_theme
        theme = self.themes.get(final_theme_key, self.themes["vivid"])
        active_pals = self.pals[theme.palette_type]

        # 3. Timeline Setup
        manual_timeline = []
        if self.timeline_script.strip():
            for line in self.timeline_script.split('\n'):
                if ':' in line:
                    try:
                        ts, st = line.split(':')
                        manual_timeline.append((float(ts.strip()), st.strip()))
                    except: continue
        
        section_styles = []
        if manual_timeline:
            manual_timeline.sort()
            # Update markers based on manual script (simplified for refactor)
            # In a full implementation, we'd sync markers to beats here
            pass

        # 4. Engine Selector Logic
        def engine_selector(t, idx, s_idx, phase):
            palette = active_pals[s_idx % 2]
            emotion = self.emotion_name if self.emotion_name != "auto" else detected_theme
            base_styles = self.emotions.get(emotion, self.categories["all"])
            
            if manual_timeline and s_idx < len(manual_timeline):
                engine_key = manual_timeline[s_idx][1]
            else:
                cat_map = self.phase_styles.get(self.category_name, self.phase_styles["all"])
                phase_options = cat_map.get(phase, cat_map[2])
                mood_styles = [s for s in base_styles if s in phase_options]
                if not mood_styles: mood_styles = phase_options
                engine_key = mood_styles[s_idx % len(mood_styles)]
            
            return self.engines.get(engine_key, self.engines["fusion"]), palette

        # 5. Render
        vw, vh = (1280, 720) if self.ratio == "16:9" else (720, 1280)
        compositor = VideoCompositor(vw, vh)
        make_frame, duration = compositor.render(data, engine_selector, self.camera, theme, self.audio_path, self.ratio)
        
        v = VideoClip(make_frame, duration=duration).with_audio(AudioFileClip(self.audio_path))
        v.write_videofile(self.out_path, fps=30, codec="libx264", audio_codec="aac")

# --- UI (Remains mostly same but cleaned up) ---
class EvolutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MV-Maker Evolution V2 (Modular)")
        self.root.geometry("500x750")
        self.file_path = tk.StringVar()
        self.category = tk.StringVar(value="all")
        self.style = tk.StringVar(value="hybrid")
        self.theme = tk.StringVar(value="auto")
        self.emotion = tk.StringVar(value="auto")
        self.ratio = tk.StringVar(value="16:9")
        
        ttk.Label(root, text="MV-Maker: MODULAR STUDIO", font=("Helvetica", 20, "bold")).pack(pady=10)
        ttk.Button(root, text="Select Audio", command=lambda: self.file_path.set(filedialog.askopenfilename())).pack(pady=5)
        ttk.Label(root, textvariable=self.file_path).pack()
        
        f0 = ttk.Frame(root); f0.pack(pady=5, fill="x", padx=20)
        ttk.Label(f0, text="Category:").pack(side="left")
        cat_combo = ttk.Combobox(f0, textvariable=self.category, values=["all", "geometric", "anime", "artistic"])
        cat_combo.pack(side="left", padx=5); cat_combo.bind("<<ComboboxSelected>>", self.update_styles)

        f1 = ttk.Frame(root); f1.pack(pady=5, fill="x", padx=20)
        ttk.Label(f1, text="Style:").pack(side="left")
        self.style_combo = ttk.Combobox(f1, textvariable=self.style)
        self.style_combo.pack(side="left", padx=5); self.update_styles()

        f2 = ttk.Frame(root); f2.pack(pady=5, fill="x", padx=20)
        ttk.Label(f2, text="Design:").pack(side="left")
        ttk.Combobox(f2, textvariable=self.theme, values=["auto", "vivid", "noir", "retro", "dreamy", "cyber", "ink", "grunge", "pop"]).pack(side="left", padx=5)

        f_emo = ttk.Frame(root); f_emo.pack(pady=5, fill="x", padx=20)
        ttk.Label(f_emo, text="Emotion:").pack(side="left")
        ttk.Combobox(f_emo, textvariable=self.emotion, values=["auto", "joy", "sadness", "anger", "peace"]).pack(side="left", padx=5)

        f3 = ttk.Frame(root); f3.pack(pady=5, fill="x", padx=20)
        ttk.Label(f3, text="Ratio:").pack(side="left")
        ttk.Combobox(f3, textvariable=self.ratio, values=["16:9", "9:16"]).pack(side="left", padx=5)

        ttk.Label(root, text="Timeline Script:").pack()
        self.script_text = tk.Text(root, height=10, width=50)
        self.script_text.pack(pady=5)
        self.script_text.insert("1.0", "0.0: slowmo\n10.0: action\n20.0: manga")
        
        ttk.Button(root, text="EVOLVE", command=self.run).pack(pady=10)

    def update_styles(self, event=None):
        dummy = MVMakerEvolution("", "")
        valid_styles = ["hybrid"] + dummy.categories.get(self.category.get(), [])
        self.style_combo['values'] = valid_styles
        if self.style.get() not in valid_styles: self.style.set("hybrid")

    def run(self):
        p = self.file_path.get()
        if p:
            out = os.path.join("output", f"{os.path.basename(p).split('.')[0]}_evolution.mp4")
            mv = MVMakerEvolution(p, out, self.style.get(), self.theme.get(), self.ratio.get(), self.script_text.get("1.0", "end"), self.category.get())
            mv.emotion_name = self.emotion.get()
            mv.create()
            messagebox.showinfo("Done", f"Evolution complete: {out}")

if __name__ == "__main__":
    if not os.path.exists("output"): os.makedirs("output")
    root = tk.Tk(); EvolutionApp(root); root.mainloop()
