import cv2
import numpy as np
import os
from style_engines import *
from design_system import DesignTheme, Camera

def generate_actual_previews():
    os.makedirs("assets/previews", exist_ok=True)
    
    # Setup dummy data for a "peak" moment
    w, h = 1280, 720
    t = 10.0
    pal = [(30, 20, 70), (130, 90, 240), (255, 190, 255)]
    intensity = 0.8
    pb = 0.9
    hm = 0.7
    sub_e = 0.5
    mel_p = [40] * 100
    mel_e = [0.8] * 100
    theme = DesignTheme("Preview", line_weight=3)
    
    engines = {
        "fusion": StyleCyberFusion(), "anime": StyleAnimePlus(), "liquid": StyleLiquidFlow(),
        "geo": StyleGeometricChaos(), "glitch": StyleGlitchPulse(), "celestial": StyleCelestialOrbit(),
        "stormy": StyleStormyLandscape(), "slowmo": StyleRainSlowMo(), "city": StyleCyberCity(),
        "graphic": StyleGraphicArt(), "ink": StyleInkWash(), "grunge": StyleGrungeGrind(), "pop": StylePopDynamic(),
        "story": StyleStorySilhouette(), "manga": StyleMangaLayout(), "action": StyleActionSilhouette(),
        "classic": StyleClassic(), "darkfantasy": StyleDarkFantasy(),
        "cyberpunk": StyleCyberPunk(), "lofi": StyleLofiChill(),
        "heavy": StyleHeavyMetal(), "synthwave": StyleSynthWave()
    }

    print("Generating actual style previews...")
    for name, engine in engines.items():
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw background style
        engine.draw(frame, t, pal, intensity, pb, hm, sub_e, mel_p, mel_e, "LR", 2, theme)
        # Draw center element
        r = int((100 + 180 * pb) * theme.line_weight / 2)
        if hasattr(engine, "draw_center"):
            engine.draw_center(frame, r, pal, theme, 2, t)
        
        out_path = f"assets/previews/{name}.png"
        cv2.imwrite(out_path, frame)
        print(f" Saved: {out_path}")

if __name__ == "__main__":
    generate_actual_previews()
