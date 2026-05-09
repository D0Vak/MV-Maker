import cv2
import numpy as np

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

        # 7. Lens Flare (Anime specific)
        if "Anime" in theme.name or theme.glow_strength > 1.2:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 3
            for i in range(1, 5):
                r_flare = int(40 * i * intensity)
                cv2.circle(frame, (cx + i*20, cy + i*10), r_flare, [int(c * 0.1 * theme.glow_strength) for c in theme.palette_type == "bright" and (255, 255, 255) or (100, 100, 255)], 1)

        return frame
