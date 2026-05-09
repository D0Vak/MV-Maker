import numpy as np
import cv2
from moviepy import VideoClip, AudioFileClip, ImageClip, CompositeVideoClip
from design_system import PostProcessor

class VideoCompositor:
    def __init__(self, vw, vh, fps=30):
        self.vw, self.vh = vw, vh
        self.fps = fps

    def render(self, audio_data, engine_selector, camera, theme, out_path, ratio):
        sr = audio_data["sr"]
        hop = 512
        duration = audio_data["duration"]
        markers = audio_data["markers"]
        section_phases = audio_data["section_phases"]
        pb, hm, sub_e, mel_p, mel_e = audio_data["pb"], audio_data["hm"], audio_data["sub_e"], audio_data["mel_p"], audio_data["mel_e"]

        def make_frame(t):
            # 1. Base Canvas (Background Layer)
            frame = np.zeros((self.vh, self.vw, 3), dtype=np.uint8)
            idx = min(int((t * sr) / hop), len(pb)-1)
            
            s_idx = 0
            for i in range(len(markers)-1):
                if markers[i] <= idx < markers[i+1]: s_idx = i; break
            
            phase = section_phases[s_idx]
            engine, palette = engine_selector(t, idx, s_idx, phase)

            # 2. Draw Style (Background/Midground)
            engine.draw(frame, t, palette, hm[idx], pb[idx], hm[idx], sub_e[idx], mel_p[max(0, idx-100):idx], mel_e[max(0, idx-100):idx], "LR", phase, theme)
            
            # 3. Camera & Effects
            is_beat = pb[idx] > 0.8 and (idx % 8 == 0)
            camera.update(pb[idx], is_beat)
            frame = camera.apply(frame)
            frame = PostProcessor.apply_effects(frame, pb[idx], hm[idx], theme)
            
            # 4. Foreground Elements (Pulsing Center etc)
            r = int(((80 + 20*phase) + 180 * pb[idx]) * theme.line_weight / 2)
            if ratio == "9:16": r = int(r * 0.7)
            
            if hasattr(engine, "draw_center"):
                engine.draw_center(frame, r, palette, theme, phase, t)
            else:
                cv2.circle(frame, (self.vw//2, self.vh//2), r, palette[2], int(theme.line_weight + phase), cv2.LINE_AA)
            
            cv2.circle(frame, (self.vw//2, self.vh//2), r + 20, (255, 255, 255), max(1, int((5 + 5*phase) * hm[idx])))
            
            # Transition flash
            if (idx - markers[s_idx]) < 10:
                flash = 1.0 - (idx - markers[s_idx]) / 10
                cv2.addWeighted(frame, 1.0, np.full_like(frame, 255), 0.4 * flash, 0, dst=frame)
            
            return frame

        # In Phase 1, we still use a single VideoClip for performance, 
        # but the structure is ready for CompositeVideoClip.
        v = VideoClip(make_frame, duration=duration).with_audio(AudioFileClip(out_path.replace("_evolution.mp4", "_temp.wav")))
        # Note: In a real multi-layer scenario, we would wrap 'v' in CompositeVideoClip([v, text_layer, logo_layer])
        
        # Temporary workaround for audio: the AudioFileClip expects the original path
        # For simplicity in this refactor, we pass the audio path directly.
        return make_frame, duration
