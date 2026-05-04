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

class DesignTheme:
    def __init__(self, name, palette_type="bright", line_weight=2, glow_strength=0.5, 
                 noise_level=0, scanlines=0, chrom_abb=0.2, saturation=1.0, contrast=1.0):
        self.name = name
        self.palette_type = palette_type
        self.line_weight = line_weight
        self.glow_strength = glow_strength
        self.noise_level = noise_level
        self.scanlines = scanlines
        self.chrom_abb = chrom_abb
        self.saturation = saturation
        self.contrast = contrast

class PostProcessor:
    @staticmethod
    def apply_effects(frame, intensity, hm, theme):
        # 1. Chromatic Aberration
        ca_strength = theme.chrom_abb * (1.0 + intensity)
        if ca_strength > 0.1:
            shift = int(10 * ca_strength)
            b, g, r = cv2.split(frame)
            b = np.roll(b, shift, axis=1)
            r = np.roll(r, -shift, axis=1)
            frame = cv2.merge([b, g, r])
        
        # 2. Bloom
        if hm > 0.7 * (1.0 / (theme.glow_strength + 1e-6)):
            blur = cv2.GaussianBlur(frame, (0, 0), 15)
            frame = cv2.addWeighted(frame, 1.0, blur, 0.4 * theme.glow_strength, 0)
            
        # 3. Saturation & Contrast
        if theme.saturation != 1.0 or theme.contrast != 1.0:
            f = frame.astype(np.float32) / 255.0
            if theme.saturation != 1.0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                gray = cv2.merge([gray, gray, gray])
                f = cv2.addWeighted(f, theme.saturation, gray, 1.0 - theme.saturation, 0)
            if theme.contrast != 1.0:
                f = np.clip((f - 0.5) * theme.contrast + 0.5, 0, 1)
            frame = (f * 255).astype(np.uint8)

        # 4. Genre Specific Textures
        if "Ink" in theme.name: # Paper Texture
            h, w = frame.shape[:2]
            paper = np.random.normal(128, 10, (h, w)).astype(np.uint8)
            paper = cv2.merge([paper, paper, paper])
            frame = cv2.addWeighted(frame, 0.9, paper, 0.1, 0)
        
        if "Grunge" in theme.name: # Dust & Scratches
            h, w = frame.shape[:2]
            for _ in range(int(5 * intensity)):
                if np.random.random() > 0.8:
                    x = np.random.randint(0, w)
                    cv2.line(frame, (x, 0), (x + np.random.randint(-10, 10), h), (255, 255, 255), 1)
                if np.random.random() > 0.9:
                    cv2.circle(frame, (np.random.randint(0, w), np.random.randint(0, h)), 2, (200, 200, 200), -1)

        # 5. Film Grain
        if theme.noise_level > 0:
            noise = np.random.normal(0, theme.noise_level, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 6. Scanlines
        if theme.scanlines > 0:
            frame[::2, :, :] = (frame[::2, :, :].astype(np.float32) * 0.8).astype(np.uint8)

        return frame

# --- Style Engines with Phase Scaling ---

class StyleCyberFusion:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.1) for c in pal[0]]
        # 1. Background Cells
        gs = 60 - (phase * 10)
        for y in range(0, h, gs):
            for x in range(0, w, gs):
                np.random.seed(x * 9 + y + int(t * 12))
                if np.random.random() > (0.96 - 0.05 * phase * intensity):
                    cv2.rectangle(frame, (x, y), (x+gs-4, y+gs-4), [int(c*0.4) for c in pal[1]], -1)
        # 2. Vibrating Electric String
        overlay = frame.copy()
        for i in range(1, len(mel_p)):
            if mel_e[i] < 0.2: continue
            vib = (10 + phase*5) * np.sin(t * 50 + i * 0.5) * mel_e[i]
            x_c, x_p = int(w * i / 100), int(w * (i-1) / 100)
            y_c, y_p = int(h * (0.1 + 0.8 * (1 - mel_p[i]/84)) + vib), int(h * (0.1 + 0.8 * (1 - mel_p[i-1]/84)) + vib)
            cv2.line(overlay, (x_p, y_p), (x_c, y_c), pal[2], int(theme.line_weight * (2 + phase)), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5 + 0.1*phase, frame, 0.5 - 0.1*phase, 0, dst=frame)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = 640, 360
        pts = []
        for i in range(6):
            a = i * np.pi / 3
            pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
        cv2.polylines(frame, [np.array(pts, np.int32)], True, pal[2], int(theme.line_weight + phase), cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(r*0.4), pal[1], -1)

class StyleAnimePlus:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        # 1. Sky
        c1, c2 = np.array(pal[1]) * 0.3, np.array(pal[0]) * 0.6
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)
        # 2. Clouds
        for i in range(3 + phase):
            np.random.seed(i * 111)
            spd = (10 + i*5) * (0.5 + 0.5*phase)
            tr = w + 500
            cx = (np.random.randint(0, tr) + int(t * spd)) % tr - 500
            cy = np.random.randint(50, h // 2)
            cv2.circle(frame, (int(cx), cy), 70, [int(c * 0.45) for c in pal[2]], -1)
            cv2.circle(frame, (int(cx)+50, cy+15), 50, [int(c * 0.45) for c in pal[2]], -1)
        # 3. Weather
        if phase >= 3 and intensity > 0.7:
            if np.random.random() > 0.96: cv2.rectangle(frame, (0,0), (w,h), (255,255,255), -1)
        if phase >= 2:
            for _ in range(int(10 * phase * theme.line_weight / 2)):
                rx, ry = (np.random.randint(0, w) + int(t*20))%w, (np.random.randint(0, h) + int(t*50))%h
                cv2.line(frame, (rx, ry), (rx-2, ry+10), (200, 200, 255), 1)
        draw_vignette(frame, 0.5 + 0.2*phase)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = 640, 360
        for i in range(12):
            a = i * np.pi / 6
            cv2.line(frame, (int(cx + r*0.8*np.cos(a)), int(cy + r*0.8*np.sin(a))), 
                     (int(cx + r*1.5*np.cos(a)), int(cy + r*1.5*np.sin(a))), pal[2], int(theme.line_weight))
        cv2.circle(frame, (cx, cy), r, pal[1], -1)

class StyleLiquidFlow:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
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
        # 2. Particles
        for i in range(int(50 * phase * theme.line_weight / 2)):
            np.random.seed(i * 123)
            px, py = (np.random.randint(0, w) + int(t*40))%w, (np.random.randint(0, h) + int(t*30))%h
            cv2.circle(frame, (px, py), int(theme.line_weight), pal[2], -1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = 640, 360
        for i in range(3):
            dr = r + int(20 * np.sin(t*5 + i))
            cv2.circle(frame, (cx + i*10, cy), dr, pal[2], int(theme.line_weight))

class StyleGeometricChaos:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
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
            cv2.polylines(frame, [np.array(pts, np.int32)], True, pal[1], int(theme.line_weight + phase), cv2.LINE_AA)
            if intensity > 0.7:
                cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c*0.2) for c in pal[2]])

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = 640, 360
        pts = []
        for i in range(3):
            a = i * 2 * np.pi / 3
            pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
        cv2.polylines(frame, [np.array(pts, np.int32)], True, pal[2], int(theme.line_weight + phase))

class StyleGlitchPulse:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.15) for c in pal[0]]
        for _ in range(int(10 * intensity * phase)):
            ry = np.random.randint(0, h)
            rh = np.random.randint(2, 30)
            cv2.rectangle(frame, (0, ry), (w, ry+rh), [int(c*0.3) for c in pal[1]], -1)
        for _ in range(int(15 * intensity * theme.line_weight / 2)):
            rx, ry = np.random.randint(0, w), np.random.randint(0, h)
            rw, rh = np.random.randint(20, 150), np.random.randint(5, 40)
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), pal[2], -1 if np.random.random() > 0.5 else 2)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = 640, 360
        cv2.rectangle(frame, (cx-r, cy-r), (cx+r, cy+r), pal[1], int(theme.line_weight))
        if np.random.random() > 0.8: cv2.rectangle(frame, (cx-r-10, cy-r+5), (cx+r+10, cy+r-5), pal[2], -1)

class StyleCelestialOrbit:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.05) for c in pal[0]]
        cx, cy = w//2, h//2
        cv2.circle(frame, (cx, cy), int(40 + 80 * pb), pal[2], -1)
        for i in range(1, 5 + phase):
            r = 80 * i
            cv2.circle(frame, (cx, cy), r, [int(c*0.4) for c in pal[1]], int(theme.line_weight), cv2.LINE_AA)
            for p in range(3 + phase):
                angle = t * (0.4 + 0.1 * i) + p * (2 * np.pi / (3 + phase))
                px, py = int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))
                cv2.circle(frame, (px, py), int(4 + 8 * hm), pal[2], -1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        pass # Built-in center

class StyleStormyLandscape:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        # Emotional Sky (Sadness/Melancholy)
        c1 = np.array(pal[0]) * 0.1
        c2 = np.array(pal[1]) * 0.2
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)
        
        # Distant Mountains
        for i in range(3):
            np.random.seed(i * 100)
            pts = [[0, h]]
            for x in range(0, w + 200, 200):
                y = h - (150 + i * 100) - np.random.randint(0, 100)
                pts.append([x, y])
            pts.append([w, h])
            cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c * (0.05 + i*0.1)) for c in pal[0]])
            
        # Rain with wind
        rain_speed = 40 + 60 * intensity
        angle = 0.1 * np.sin(t * 0.5)
        for i in range(int(150 * theme.line_weight / 2)):
            np.random.seed(i * 99)
            rx, ry = (np.random.randint(0, w) + int(t * 20)) % w, (np.random.randint(0, h) + int(t * rain_speed)) % h
            cv2.line(frame, (rx, ry), (int(rx + 30*angle), ry + 30), (160, 160, 200), 1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        cv2.circle(frame, (cx, cy), r, (100, 100, 150), 1)

class StyleStorySilhouette:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        # 1. Sky Gradient (Changes with phase for story progression)
        if phase == 1: # Dawn/Intro
            c1, c2 = np.array([20, 20, 40]), np.array([100, 60, 40])
        elif phase == 2: # Day/Progress
            c1, c2 = np.array([150, 100, 40]), np.array([40, 40, 80])
        else: # Night/Climax
            c1, c2 = np.array([10, 10, 20]), np.array([30, 20, 60])
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)

        # 2. Distant City/Mountain (Slow scroll)
        for i in range(2):
            np.random.seed(i * 555)
            pts = [[0, h]]
            off = int(t * (10 + i*10)) % 400
            for x in range(-off, w + 400, 400):
                y = h - (100 + i*80) - np.random.randint(0, 50)
                pts.append([x, y])
            pts.append([w, h])
            cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c * (0.1 + i*0.1)) for c in pal[0]])

        # 3. Road & Ground
        ground_y = int(h * 0.8)
        cv2.rectangle(frame, (0, ground_y), (w, h), [int(c * 0.05) for c in pal[0]], -1)

        # 4. Moving Silhouettes (Power poles, trees)
        pole_spacing = 600
        pole_offset = int(t * 300) % pole_spacing
        for x in range(w + pole_spacing, -pole_spacing, -pole_spacing):
            px = x - pole_offset
            if px < -50 or px > w + 50: continue
            # Pole
            cv2.rectangle(frame, (px, ground_y - 400), (px + 15, ground_y), (5, 5, 10), -1)
            # Wires
            cv2.line(frame, (px, ground_y - 380), (px + pole_spacing, ground_y - 360), (10, 10, 15), 1)

        # 5. Sun/Moon (Audio reactive)
        cel_y = int(h * 0.3 - 100 * np.sin(t * 0.1))
        color = (255, 255, 255) if phase == 3 else (200, 220, 255)
        r_cel = int(50 + 50 * pb)
        cv2.circle(frame, (w - 200, cel_y), r_cel, color, -1)
        if phase == 3: # Moon Glow
            cv2.circle(frame, (w - 200, cel_y), r_cel + 30, color, 1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        pass # Story style focuses on the scene, no abstract center

class StyleRainSlowMo:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.1) for c in pal[0]]
        for i in range(10 + 5 * phase):
            np.random.seed(i * 88)
            rx = np.random.randint(0, w)
            ry = (np.random.randint(0, h) + int(t * 30)) % h
            size = int((3 + 5 * intensity) * theme.line_weight / 2)
            cv2.circle(frame, (rx, ry), size, (200, 200, 255), -1)
            cv2.circle(frame, (rx, ry), size + 4, (100, 100, 150), 1)
        for i in range(3):
            np.random.seed(i * 22 + int(t * 2))
            rx, ry = np.random.randint(0, w), h - np.random.randint(0, 50)
            r_ripple = int((t % 1) * 120 * intensity)
            cv2.ellipse(frame, (rx, ry), (r_ripple, r_ripple//4), 0, 0, 360, (150, 150, 200), int(theme.line_weight))

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = 640, 360
        cv2.circle(frame, (cx, cy), r, pal[2], 1)
        cv2.circle(frame, (cx, cy), r+10, pal[1], 1)

class StyleCyberCity:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.05) for c in pal[0]]
        for i in range(15):
            np.random.seed(i * 33)
            bw = int(w * 0.1)
            bx = i * bw - bw
            bounce = int(50 * pb * np.sin(t * 2 + i))
            bh = int(h * 0.3) + np.random.randint(0, h // 2) + bounce
            cv2.rectangle(frame, (bx, h - bh), (bx + bw, h), [int(c * (0.1 + 0.05*i)) for c in pal[1]], -1)
            if intensity > 0.4:
                for wy in range(h - bh + 20, h, 40):
                    for wx in range(bx + 15, bx + bw - 15, 25):
                        if np.random.random() > 0.6:
                            c_win = pal[2] if np.random.random() > 0.8 else [int(c*0.4) for c in pal[2]]
                            cv2.rectangle(frame, (wx, wy), (wx+10, wy+12), c_win, -1)
        for i in range(3):
            np.random.seed(i * 999)
            if np.random.random() > 0.7:
                nx, ny = np.random.randint(0, w), np.random.randint(100, h-200)
                cv2.rectangle(frame, (nx, ny), (nx+60, ny+20), pal[2], 2)

    def draw_center(self, frame, r, pal, theme, phase, t):
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        cv2.line(frame, (cx-r, cy), (cx+r, cy), pal[2], 1)
        cv2.line(frame, (cx, cy-r), (cx, cy+r), pal[2], 1)
        cv2.circle(frame, (cx, cy), r, pal[2], 1)

class StyleInkWash:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.9) for c in pal[0]] # Bright paper
        for i in range(2 + phase):
            np.random.seed(i * 123)
            cx, cy = w//2 + int(w*0.2 * np.cos(t*0.2 + i)), h//2 + int(h*0.2 * np.sin(t*0.3 + i))
            r = int((100 + 100 * intensity) * (1 + 0.2 * np.sin(t*2)))
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), r, pal[1], -1)
            cv2.circle(overlay, (cx+20, cy-10), r//2, pal[2], -1)
            # Bleed effect
            overlay = cv2.GaussianBlur(overlay, (0, 0), 20 + 10 * intensity)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)

    def draw_center(self, frame, r, pal, theme, phase, t):
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        cv2.circle(frame, (cx, cy), int(r * 0.8), [int(c*0.5) for c in pal[1]], 2)
        cv2.circle(frame, (cx, cy), int(r * 0.4), pal[2], -1)

class StyleGrungeGrind:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.05) for c in pal[0]]
        # Jagged lines
        for i in range(10 + 20 * phase):
            np.random.seed(i + int(t * 15))
            pts = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(3)], np.int32)
            cv2.polylines(frame, [pts], False, pal[1], int(theme.line_weight + 2 * intensity))
        # High contrast noise
        if pb > 0.8:
            overlay = np.random.randint(0, 255, frame.shape, dtype=np.uint8)
            cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, dst=frame)

    def draw_center(self, frame, r, pal, theme, phase, t):
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        for _ in range(5):
            pts = np.array([[cx + np.random.randint(-r, r), cy + np.random.randint(-r, r)] for _ in range(3)], np.int32)
            cv2.fillPoly(frame, [pts], pal[2])

class StylePopDynamic:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = pal[0]
        # Bouncy circles
        for i in range(5 + 5 * phase):
            np.random.seed(i * 44)
            bounce = abs(np.sin(t * 4 + i)) * 100 * intensity
            cx = int(w * (i+1) / 10)
            cy = int(h * 0.8 - bounce)
            cv2.circle(frame, (cx, cy), int(40 * (1 + intensity)), pal[1], -1)
            cv2.circle(frame, (cx, cy), int(50 * (1 + intensity)), pal[2], int(theme.line_weight))
        # Text deco
        cv2.putText(frame, "POP_SYNC", (w-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, pal[2], 3)

    def draw_center(self, frame, r, pal, theme, phase, t):
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        s = int(r * 1.2)
        cv2.rectangle(frame, (cx-s, cy-s), (cx+s, cy+s), pal[2], int(theme.line_weight * 2))
        cv2.putText(frame, "HIT!", (cx-20, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 1, pal[1], 4)

class StyleGraphicArt:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.95) for c in pal[0]] # Bright paper background
        # 1. Floating Dots
        for i in range(20 + 20 * phase):
            np.random.seed(i * 555)
            x, y = (np.random.randint(0, w) + int(t * 10)) % w, np.random.randint(0, h)
            cv2.circle(frame, (x, y), 3, [int(c*0.3) for c in pal[1]], -1)
        # 2. Big Graphic Elements
        for i in range(2):
            np.random.seed(i * 123)
            r_big = 200 + 100 * intensity
            cv2.circle(frame, (w, 0), int(r_big), [int(c*0.2) for c in pal[2]], int(theme.line_weight * 5))
            cv2.circle(frame, (0, h), int(r_big * 0.5), [int(c*0.2) for c in pal[1]], -1)
        # 3. Dynamic Bars
        for i in range(10):
            bh = int(100 * pb * (1 + i*0.1))
            cv2.rectangle(frame, (100 + i*20, h-50-bh), (110 + i*20, h-50), pal[2], -1)
        # 4. Text-like deco
        cv2.putText(frame, "AUDIO_SYNC_V2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [int(c*0.3) for c in pal[0]], 2)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = 640, 360
        for i in range(3):
            cv2.circle(frame, (cx, cy), r - i*20, pal[1], int(theme.line_weight))
        cv2.line(frame, (cx-r-20, cy), (cx+r+20, cy), pal[2], 2)
        cv2.line(frame, (cx, cy-r-20), (cx, cy+r+20), pal[2], 2)

# --- Engine ---

class MVMakerEvolution:
    def __init__(self, audio_path, out_path, style_name="hybrid", theme_name="auto", ratio="16:9", timeline_script=""):
        self.audio_path, self.out_path, self.style_name, self.theme_name, self.emotion_name, self.ratio, self.timeline_script = \
            audio_path, out_path, style_name, theme_name, "auto", ratio, timeline_script
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
            "story": StyleStorySilhouette(), "melancholy": StyleStormyLandscape()
        }
        self.emotions = {
            "joy": ["pop", "anime", "graphic"],
            "sadness": ["melancholy", "ink", "slowmo"],
            "anger": ["grunge", "glitch", "geo"],
            "peace": ["celestial", "story", "liquid"]
        }
        self.camera = Camera()
        
    def create(self):
        print("Starting Evolutionary Analysis...")
        # Robust audio loading (supports .m4a, .mp3 etc using moviepy/ffmpeg fallback)
        try:
            y, sr = librosa.load(self.audio_path, sr=None)
        except:
            print("Librosa load failed, falling back to MoviePy decoder...")
            audio_tmp = AudioFileClip(self.audio_path)
            sr = 44100
            y = audio_tmp.to_soundarray(fps=sr)
            if y.ndim > 1: y = y.mean(axis=1) # Mono conversion
            audio_tmp.close()

        duration = librosa.get_duration(y=y, sr=sr)
        hop = 512
        
        # 1. Genre & Mood Analysis
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        tempo = float(np.mean(tempo))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y, hop_length=hop))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=hop))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        detected_theme = "vivid"
        if zcr > 0.08 or flatness > 0.04: # Aggressive/Rock
            detected_theme = "grunge"
        elif rolloff < 3000 and flatness < 0.01: # Mellow/Jazz
            detected_theme = "ink"
        elif tempo > 120 and flatness > 0.02: # Pop
            detected_theme = "pop"
        elif tempo < 90:
            detected_theme = "dreamy"
        else:
            detected_theme = "vivid"
            
        final_theme_key = self.theme_name if self.theme_name != "auto" else detected_theme
        theme = self.themes.get(final_theme_key, self.themes["vivid"])
        print(f"Genre Analysis -> Tempo: {tempo:.1f}, ZCR: {zcr:.4f}, Flatness: {flatness:.4f}")
        print(f"Applying Theme: {theme.name}")

        # 2. Structural Segmentation
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
        
        if manual_timeline:
            manual_timeline.sort()
            for ts, st in manual_timeline:
                f_idx = int((ts * sr) / hop)
                if len(beats) > 0:
                    q_idx = beats[np.abs(beats - f_idx).argmin()]
                    markers.append(q_idx)
                else: markers.append(f_idx)
                section_styles.append(st)
        else:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop)
            bounds = librosa.segment.agglomerative(S, k=min(14, int(duration/8)))
            if len(beats) > 0:
                for bf in bounds:
                    q_idx = round(np.abs(beats - bf).argmin() / 16) * 16
                    if q_idx < len(beats): markers.append(beats[q_idx])
            
        markers.append(int((duration * sr) / hop))
        markers = sorted(list(set(markers)))
        
        # 3. Phase Calculation
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        sec_energies = [np.mean(rms[markers[i]:markers[i+1]]) for i in range(len(markers)-1)]
        sorted_e = sorted(sec_energies)
        p1_t, p2_t = sorted_e[len(sorted_e)//3], sorted_e[2*len(sorted_e)//3]
        section_phases = [(1 if e <= p1_t else 2 if e <= p2_t else 3) for e in sec_energies]
        
        # 4. Other Features
        active_pals = self.pals[theme.palette_type]
        y_h, y_p = librosa.effects.hpss(y)
        pb = np.mean(np.clip((librosa.power_to_db(librosa.feature.melspectrogram(y=y_p, sr=sr, n_mels=128, hop_length=hop), ref=np.max)+80)/80,0,1)[0:20,:], axis=0)
        pb = (pb / (np.max(pb)+1e-6))**2
        hm = np.convolve(np.mean(np.clip((librosa.power_to_db(librosa.feature.melspectrogram(y=y_h, sr=sr, n_mels=128, hop_length=hop), ref=np.max)+80)/80,0,1)[10:80,:], axis=0), np.ones(10)/10, mode='same')
        cqt_n = np.clip((librosa.amplitude_to_db(np.abs(librosa.cqt(y_h, sr=sr, hop_length=hop, n_bins=84)), ref=np.max)+80)/80,0,1)
        mel_p, mel_e = np.argmax(cqt_n, axis=0), np.max(cqt_n, axis=0)
        sub_e = np.max(cqt_n.copy(), axis=0)

        # 6. Rendering Setup
        vw, vh = (1280, 720) if self.ratio == "16:9" else (720, 1280)
        print(f"Rendering Evolution ({self.ratio})...")
        def make_frame(t):
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)
            idx = min(int((t * sr) / hop), len(pb)-1)
            s_idx = 0
            for i in range(len(markers)-1):
                if markers[i] <= idx < markers[i+1]: s_idx = i; break
            
            phase = section_phases[s_idx]
            palette = active_pals[s_idx % 2]
            
            # Emotional Selection
            emotion = self.emotion_name if self.emotion_name != "auto" else detected_theme # Reusing detection for now
            mood_styles = self.emotions.get(emotion, list(self.engines.keys()))
            
            if manual_timeline:
                engine_key = section_styles[s_idx] if s_idx < len(section_styles) else section_styles[-1]
            else:
                if self.style_name == "hybrid":
                    engine_key = mood_styles[s_idx % len(mood_styles)]
                else:
                    engine_key = self.style_name
            engine = self.engines.get(engine_key, self.engines["fusion"])
            
            # Draw style
            engine.draw(frame, t, palette, hm[idx], pb[idx], hm[idx], sub_e[idx], mel_p[max(0, idx-100):idx], mel_e[max(0, idx-100):idx], "LR", phase, theme)
            
            # Camera & Effects
            is_beat = pb[idx] > 0.8 and (idx % 8 == 0)
            self.camera.update(pb[idx], is_beat)
            frame = self.camera.apply(frame)
            frame = PostProcessor.apply_effects(frame, pb[idx], hm[idx], theme)
            
            # Pulsing Center
            r = int(((80 + 20*phase) + 180 * pb[idx]) * theme.line_weight / 2)
            if self.ratio == "9:16": r = int(r * 0.7)
            
            if hasattr(engine, "draw_center"):
                engine.draw_center(frame, r, palette, theme, phase, t)
            else:
                cv2.circle(frame, (vw//2, vh//2), r, palette[2], int(theme.line_weight + phase), cv2.LINE_AA)
            
            # Sub circle for all
            cv2.circle(frame, (vw//2, vh//2), r + 20, (255, 255, 255), max(1, int((5 + 5*phase) * hm[idx])))
            
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
        self.root.geometry("500x650")
        self.file_path = tk.StringVar()
        self.style = tk.StringVar(value="hybrid")
        self.theme = tk.StringVar(value="auto")
        self.emotion = tk.StringVar(value="auto")
        self.ratio = tk.StringVar(value="16:9")
        ttk.Label(root, text="MV-Maker: EVOLUTION", font=("Helvetica", 20, "bold")).pack(pady=10)
        ttk.Button(root, text="Select Audio", command=lambda: self.file_path.set(filedialog.askopenfilename())).pack(pady=5)
        ttk.Label(root, textvariable=self.file_path).pack()
        
        f1 = ttk.Frame(root)
        f1.pack(pady=5, fill="x", padx=20)
        ttk.Label(f1, text="Style:").pack(side="left")
        ttk.Combobox(f1, textvariable=self.style, values=["hybrid", "fusion", "anime", "liquid", "geo", "glitch", "celestial", "stormy", "slowmo", "city", "graphic", "ink", "grunge", "pop", "story", "melancholy"]).pack(side="left", padx=5)
        
        f2 = ttk.Frame(root)
        f2.pack(pady=5, fill="x", padx=20)
        ttk.Label(f2, text="Design:").pack(side="left")
        ttk.Combobox(f2, textvariable=self.theme, values=["auto", "vivid", "noir", "retro", "dreamy", "cyber", "ink", "grunge", "pop"]).pack(side="left", padx=5)

        f_emo = ttk.Frame(root)
        f_emo.pack(pady=5, fill="x", padx=20)
        ttk.Label(f_emo, text="Emotion:").pack(side="left")
        ttk.Combobox(f_emo, textvariable=self.emotion, values=["auto", "joy", "sadness", "anger", "peace"]).pack(side="left", padx=5)

        f3 = ttk.Frame(root)
        f3.pack(pady=5, fill="x", padx=20)
        ttk.Label(f3, text="Ratio:").pack(side="left")
        ttk.Combobox(f3, textvariable=self.ratio, values=["16:9", "9:16"]).pack(side="left", padx=5)

        ttk.Label(root, text="Timeline Script (time: style):").pack()
        self.script_text = tk.Text(root, height=10, width=50)
        self.script_text.pack(pady=5)
        self.script_text.insert("1.0", "0.0: slowmo\n10.0: stormy\n20.0: city")
        
        ttk.Button(root, text="EVOLVE", command=self.run).pack(pady=10)
    def run(self):
        p = self.file_path.get()
        if p:
            out = os.path.join("output", f"{os.path.basename(p).split('.')[0]}_evolution.mp4")
            mv = MVMakerEvolution(p, out, self.style.get(), self.theme.get(), self.ratio.get(), self.script_text.get("1.0", "end"))
            mv.emotion_name = self.emotion.get()
            mv.create()
            messagebox.showinfo("Done", f"Evolution complete: {out}")

if __name__ == "__main__":
    if not os.path.exists("output"): os.makedirs("output")
    root = tk.Tk()
    EvolutionApp(root)
    root.mainloop()
