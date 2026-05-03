import librosa
import numpy as np
import cv2
from moviepy import VideoClip, AudioFileClip
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# --- Global Helpers ---
def draw_vignette(frame, strength=1.0):
    h, w = frame.shape[:2]
    k_x, k_y = cv2.getGaussianKernel(w, w/2), cv2.getGaussianKernel(h, h/2)
    mask = (k_y * k_x.T) / (k_y * k_x.T).max()
    mask = 1 - (1 - mask) * strength
    for i in range(3): frame[:, :, i] = (frame[:, :, i] * mask).astype(np.uint8)

class Camera:
    def __init__(self):
        self.offset = [0, 0]
        self.zoom = 1.0
    
    def update(self, intensity, is_beat):
        if is_beat:
            self.offset = [np.random.randint(-15, 15) * intensity, np.random.randint(-15, 15) * intensity]
            self.zoom = 1.0 + 0.05 * intensity
        else:
            self.offset = [o * 0.85 for o in self.offset]
            self.zoom = 1.0 + (self.zoom - 1.0) * 0.85
            
    def apply(self, frame):
        h, w = frame.shape[:2]
        M = np.float32([[self.zoom, 0, self.offset[0] + (1-self.zoom)*w/2], [0, self.zoom, self.offset[1] + (1-self.zoom)*h/2]])
        return cv2.warpAffine(frame, M, (w, h))

class PostProcessor:
    @staticmethod
    def apply_effects(frame, intensity, hm):
        # Chromatic Aberration
        if intensity > 0.6:
            shift = int(10 * intensity)
            b, g, r = cv2.split(frame)
            b = np.roll(b, shift, axis=1)
            r = np.roll(r, -shift, axis=1)
            frame = cv2.merge([b, g, r])
        # Bloom
        if hm > 0.7:
            blur = cv2.GaussianBlur(frame, (0, 0), 15)
            frame = cv2.addWeighted(frame, 1.0, blur, 0.4, 0)
        return frame

# --- Style Engines with Phase Scaling ---

class StyleCyberFusion:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.1) for c in pal[0]]
        # 1. Background Cells (scaled by phase)
        gs = 60 - (phase * 10)
        for y in range(0, h, gs):
            for x in range(0, w, gs):
                np.random.seed(x * 9 + y + int(t * 12))
                if np.random.random() > (0.96 - 0.05 * phase * intensity):
                    cv2.rectangle(frame, (x, y), (x+gs-4, y+gs-4), [int(c*0.4) for c in pal[1]], -1)
        # 2. Vibrating Electric String (thickness scaled by phase)
        overlay = frame.copy()
        for i in range(1, len(mel_p)):
            if mel_e[i] < 0.2: continue
            vib = (10 + phase*5) * np.sin(t * 50 + i * 0.5) * mel_e[i]
            x_c, x_p = int(w * i / 100), int(w * (i-1) / 100)
            y_c, y_p = int(h * (0.1 + 0.8 * (1 - mel_p[i]/84)) + vib), int(h * (0.1 + 0.8 * (1 - mel_p[i-1]/84)) + vib)
            cv2.line(overlay, (x_p, y_p), (x_c, y_c), pal[2], 5 + phase*3, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5 + 0.1*phase, frame, 0.5 - 0.1*phase, 0, dst=frame)

class StyleAnimePlus:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        # 1. Sky
        c1, c2 = np.array(pal[1]) * 0.3, np.array(pal[0]) * 0.6
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)
        # 2. Clouds (Speed scaled by phase)
        for i in range(3 + phase):
            np.random.seed(i * 111)
            spd = (10 + i*5) * (0.5 + 0.5*phase)
            tr = w + 500
            cx = (np.random.randint(0, tr) + int(t * spd)) % tr - 500
            cy = np.random.randint(50, h // 2)
            cv2.circle(frame, (int(cx), cy), 70, [int(c * 0.45) for c in pal[2]], -1)
            cv2.circle(frame, (int(cx)+50, cy+15), 50, [int(c * 0.45) for c in pal[2]], -1)
        # 3. Weather (Complexity scaled by phase)
        if phase >= 3 and intensity > 0.7: # Thunder
            if np.random.random() > 0.96: cv2.rectangle(frame, (0,0), (w,h), (255,255,255), -1)
        if phase >= 2: # Particles
            for _ in range(10 * phase):
                rx, ry = (np.random.randint(0, w) + int(t*20))%w, (np.random.randint(0, h) + int(t*50))%h
                cv2.line(frame, (rx, ry), (rx-2, ry+10), (200, 200, 255), 1)
        draw_vignette(frame, 0.5 + 0.2*phase)

class StyleLiquidFlow:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.2) for c in pal[0]]
        # 1. Morphing Blobs
        for i in range(2 + phase):
            np.random.seed(i * 99)
            cx, cy = w/2 + w*0.2 * np.cos(t * 0.4 + i), h/2 + h*0.2 * np.sin(t * 0.6 + i)
            pts = []
            for v in range(12):
                a = 2 * np.pi * v / 12
                r = 120 + (50 + 20*phase) * np.sin(t * 2 + v + i) * intensity
                pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
            cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c * 0.4 * intensity) for c in pal[1]])
        # 2. High Density Particles (Scaled by phase)
        for i in range(50 * phase):
            np.random.seed(i * 123)
            px, py = (np.random.randint(0, w) + int(t*40))%w, (np.random.randint(0, h) + int(t*30))%h
            cv2.circle(frame, (px, py), 2, pal[2], -1)

class StyleGeometricChaos:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.1) for c in pal[0]]
        cx, cy = w//2, h//2
        for i in range(5 + phase * 2):
            np.random.seed(i * 44)
            size = (100 + 50 * phase) * intensity + np.random.randint(20, 150)
            angle = t * (1 + i * 0.3)
            pts = []
            sides = 3 + (i % 4)
            for s in range(sides):
                a = angle + 2 * np.pi * s / sides
                pts.append([int(cx + size * np.cos(a)), int(cy + size * np.sin(a))])
            cv2.polylines(frame, [np.array(pts, np.int32)], True, pal[1], 2 + phase, cv2.LINE_AA)
            if intensity > 0.7:
                cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c*0.2) for c in pal[2]])

class StyleGlitchPulse:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.15) for c in pal[0]]
        for _ in range(int(10 * intensity * phase)):
            ry = np.random.randint(0, h)
            rh = np.random.randint(2, 30)
            cv2.rectangle(frame, (0, ry), (w, ry+rh), [int(c*0.3) for c in pal[1]], -1)
        for _ in range(int(15 * intensity)):
            rx, ry = np.random.randint(0, w), np.random.randint(0, h)
            rw, rh = np.random.randint(20, 150), np.random.randint(5, 40)
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), pal[2], -1 if np.random.random() > 0.5 else 2)

class StyleCelestialOrbit:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.05) for c in pal[0]]
        cx, cy = w//2, h//2
        cv2.circle(frame, (cx, cy), int(40 + 80 * pb), pal[2], -1)
        for i in range(1, 5 + phase):
            r = 80 * i
            cv2.circle(frame, (cx, cy), r, [int(c*0.4) for c in pal[1]], 1, cv2.LINE_AA)
            for p in range(3 + phase):
                angle = t * (0.4 + 0.1 * i) + p * (2 * np.pi / (3 + phase))
                px, py = int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))
                cv2.circle(frame, (px, py), int(4 + 8 * hm), pal[2], -1)

class StyleStormyLandscape:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        c1, c2 = np.array(pal[0]) * 0.2, np.array(pal[1]) * 0.1
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)
        for i in range(2):
            np.random.seed(i * 50)
            pts = [[0, h]]
            for x in range(0, w + 120, 120):
                y = h - (100 + i * 120) - np.random.randint(0, 60)
                pts.append([x, y])
            pts.append([w, h])
            cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c * (0.1 + i*0.1)) for c in pal[0]])
        for _ in range(50 + 50 * phase):
            rx, ry = np.random.randint(0, w), np.random.randint(0, h)
            len_r = 20 + 30 * intensity
            cv2.line(frame, (rx, ry), (rx - 5, ry + int(len_r)), (180, 180, 220), 1)
        if pb > 0.9 and np.random.random() > 0.8:
            cv2.rectangle(frame, (0, 0), (w, h), (255, 255, 255), -1)

class StyleRainSlowMo:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.1) for c in pal[0]]
        for i in range(10 + 5 * phase):
            np.random.seed(i * 88)
            rx = np.random.randint(0, w)
            ry = (np.random.randint(0, h) + int(t * 30)) % h
            size = int(3 + 5 * intensity)
            cv2.circle(frame, (rx, ry), size, (200, 200, 255), -1)
            cv2.circle(frame, (rx, ry), size + 4, (100, 100, 150), 1)
        for i in range(3):
            np.random.seed(i * 22 + int(t * 2))
            rx, ry = np.random.randint(0, w), h - np.random.randint(0, 50)
            r = int((t % 1) * 120 * intensity)
            cv2.ellipse(frame, (rx, ry), (r, r//4), 0, 0, 360, (150, 150, 200), 1)

class StyleCyberCity:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.05) for c in pal[0]]
        for i in range(12):
            np.random.seed(i * 33)
            bw = 70 + np.random.randint(0, 50)
            bx = i * 110
            bh = 150 + np.random.randint(0, 350)
            cv2.rectangle(frame, (bx, h - bh), (bx + bw, h), [int(c * 0.1) for c in pal[1]], -1)
            if intensity > 0.4:
                for wy in range(h - bh + 20, h, 40):
                    for wx in range(bx + 15, bx + bw - 15, 25):
                        if np.random.random() > 0.6:
                            cv2.rectangle(frame, (wx, wy), (wx+10, wy+12), pal[2] if np.random.random() > 0.7 else [int(c*0.4) for c in pal[2]], -1)

# --- Engine ---

class MVMakerEvolution:
    def __init__(self, audio_path, out_path, style_name="hybrid", timeline_script=""):
        self.audio_path, self.out_path, self.style_name, self.timeline_script = audio_path, out_path, style_name, timeline_script
        self.pals = {
            "bright": [[(30, 20, 70), (130, 90, 240), (255, 190, 255)], [(15, 45, 55), (40, 200, 140), (160, 255, 240)]],
            "deep": [[(15, 10, 40), (70, 40, 140), (160, 90, 255)], [(30, 15, 20), (160, 70, 40), (255, 130, 90)]]
        }
        self.engines = {
            "fusion": StyleCyberFusion(), "anime": StyleAnimePlus(), "liquid": StyleLiquidFlow(),
            "geo": StyleGeometricChaos(), "glitch": StyleGlitchPulse(), "celestial": StyleCelestialOrbit(),
            "stormy": StyleStormyLandscape(), "slowmo": StyleRainSlowMo(), "city": StyleCyberCity()
        }
        self.camera = Camera()
        
    def create(self):
        print("Starting Evolutionary Analysis...")
        y, sr = librosa.load(self.audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        hop = 512
        
        # 1. Structural Segmentation
        markers = [0]
        section_styles = []
        
        manual_timeline = []
        if self.timeline_script.strip():
            for line in self.timeline_script.split('\n'):
                if ':' in line:
                    try:
                        ts, st = line.split(':')
                        manual_timeline.append((float(ts.strip()), st.strip()))
                    except: continue
        
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        
        if manual_timeline:
            manual_timeline.sort()
            for ts, st in manual_timeline:
                f_idx = int((ts * sr) / hop)
                if len(beats) > 0:
                    q_idx = beats[np.abs(beats - f_idx).argmin()]
                    markers.append(q_idx)
                else:
                    markers.append(f_idx)
                section_styles.append(st)
        else:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop)
            bounds = librosa.segment.agglomerative(S, k=min(14, int(duration/8)))
            if len(beats) > 0:
                for bf in bounds:
                    q_idx = round(np.abs(beats - bf).argmin() / 16) * 16
                    if q_idx < len(beats): markers.append(beats[q_idx])
            
        markers.append(S.shape[1] if not manual_timeline else int((duration * sr) / hop))
        markers = sorted(list(set(markers)))
        
        # 2. Phase Calculation (Categorize sections into 3 phases by energy)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        sec_energies = [np.mean(rms[markers[i]:markers[i+1]]) for i in range(len(markers)-1)]
        sorted_e = sorted(sec_energies)
        p1_t, p2_t = sorted_e[len(sorted_e)//3], sorted_e[2*len(sorted_e)//3]
        section_phases = [(1 if e <= p1_t else 2 if e <= p2_t else 3) for e in sec_energies]
        
        # 3. Features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        active_pals = self.pals["bright" if centroid > 2600 else "deep"]
        y_h, y_p = librosa.effects.hpss(y)
        pb = np.mean(np.clip((librosa.power_to_db(librosa.feature.melspectrogram(y=y_p, sr=sr, n_mels=128, hop_length=hop), ref=np.max)+80)/80,0,1)[0:20,:], axis=0)
        pb = (pb / (np.max(pb)+1e-6))**2
        hm = np.convolve(np.mean(np.clip((librosa.power_to_db(librosa.feature.melspectrogram(y=y_h, sr=sr, n_mels=128, hop_length=hop), ref=np.max)+80)/80,0,1)[10:80,:], axis=0), np.ones(10)/10, mode='same')
        cqt_n = np.clip((librosa.amplitude_to_db(np.abs(librosa.cqt(y_h, sr=sr, hop_length=hop, n_bins=84)), ref=np.max)+80)/80,0,1)
        mel_p, mel_e = np.argmax(cqt_n, axis=0), np.max(cqt_n, axis=0)
        sub_e = np.max(cqt_n.copy(), axis=0)

        print("Rendering Evolution...")
        def make_frame(t):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            idx = min(int((t * sr) / hop), len(pb)-1)
            s_idx = 0
            for i in range(len(markers)-1):
                if markers[i] <= idx < markers[i+1]: s_idx = i; break
            
            phase = section_phases[s_idx]
            palette = active_pals[s_idx % 2]
            keys = list(self.engines.keys())
            if manual_timeline:
                engine_key = section_styles[s_idx] if s_idx < len(section_styles) else section_styles[-1]
            else:
                engine_key = keys[s_idx % len(keys)] if self.style_name == "hybrid" else self.style_name
            engine = self.engines.get(engine_key, self.engines["fusion"])
            
            # Draw style
            engine.draw(frame, t, palette, hm[idx], pb[idx], hm[idx], sub_e[idx], mel_p[max(0, idx-100):idx], mel_e[max(0, idx-100):idx], "LR", phase)
            
            # Camera & Effects
            is_beat = pb[idx] > 0.8 and (idx % 8 == 0)
            self.camera.update(pb[idx], is_beat)
            frame = self.camera.apply(frame)
            frame = PostProcessor.apply_effects(frame, pb[idx], hm[idx])
            
            # Pulsing Center (Reduced scale if celestial)
            if engine_key != "celestial":
                r = int((80 + 20*phase) + 180 * pb[idx])
                cv2.circle(frame, (640, 360), r, palette[2], 3 + 2*phase, cv2.LINE_AA)
                cv2.circle(frame, (640, 360), r + 20, (255, 255, 255), max(1, int((5 + 5*phase) * hm[idx])))
            
            # Transition flash
            if (idx - markers[s_idx]) < 10:
                flash = 1.0 - (idx - markers[s_idx]) / 10
                cv2.addWeighted(frame, 1.0, np.full_like(frame, 255), 0.4 * flash, 0, dst=frame)
            
            return frame

        v = VideoClip(make_frame, duration=duration).with_audio(AudioFileClip(self.audio_path))
        v.write_videofile(self.out_path, fps=30, codec="libx264", audio_codec="aac")

# --- UI ---
class EvolutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MV-Maker Evolution")
        self.root.geometry("500x550")
        self.file_path = tk.StringVar()
        self.style = tk.StringVar(value="hybrid")
        ttk.Label(root, text="MV-Maker: EVOLUTION", font=("Helvetica", 20, "bold")).pack(pady=10)
        ttk.Button(root, text="Select Audio", command=lambda: self.file_path.set(filedialog.askopenfilename())).pack(pady=5)
        ttk.Label(root, textvariable=self.file_path).pack()
        
        frame = ttk.Frame(root)
        frame.pack(pady=5, fill="x", padx=20)
        ttk.Label(frame, text="Style:").pack(side="left")
        ttk.Combobox(frame, textvariable=self.style, values=["hybrid", "fusion", "anime", "liquid", "geo", "glitch", "celestial", "stormy", "slowmo", "city"]).pack(side="left", padx=5)
        
        ttk.Label(root, text="Timeline Script (time: style):").pack()
        self.script_text = tk.Text(root, height=10, width=50)
        self.script_text.pack(pady=5)
        self.script_text.insert("1.0", "0.0: slowmo\n10.0: stormy\n20.0: city")
        
        ttk.Button(root, text="EVOLVE", command=self.run).pack(pady=10)
    def run(self):
        p = self.file_path.get()
        if p:
            out = os.path.join("output", f"{os.path.basename(p).split('.')[0]}_evolution.mp4")
            MVMakerEvolution(p, out, self.style.get(), self.script_text.get("1.0", "end")).create()
            messagebox.showinfo("Done", f"Evolution complete: {out}")

if __name__ == "__main__":
    if not os.path.exists("output"): os.makedirs("output")
    root = tk.Tk()
    EvolutionApp(root)
    root.mainloop()
