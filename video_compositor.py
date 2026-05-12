import numpy as np
import cv2
from moviepy import VideoClip, AudioFileClip, ImageClip, CompositeVideoClip
from design_system import PostProcessor
from PIL import Image, ImageDraw, ImageFont
import os

class VideoCompositor:
    def __init__(self, vw, vh, fps=30):
        self.vw, self.vh = vw, vh
        self.fps = fps
        # Try to find a Japanese font for Windows
        self.font_path = "C:\\Windows\\Fonts\\msgothic.ttc"
        if not os.path.exists(self.font_path):
            self.font_path = None # Fallback

    def render(self, audio_data, engine_selector, camera, theme, out_path, ratio, lyrics_segments=None, base_visual_path=""):
        sr = audio_data["sr"]
        hop = 512
        duration = audio_data["duration"]
        markers = audio_data["markers"]
        section_phases = audio_data["section_phases"]
        pb, hm, sub_e, mel_p, mel_e = audio_data["pb"], audio_data["hm"], audio_data["sub_e"], audio_data["mel_p"], audio_data["mel_e"]

        base_clip = None
        if base_visual_path and os.path.exists(base_visual_path):
            try:
                base_clip = VideoFileClip(base_visual_path)
            except:
                try:
                    base_clip = ImageClip(base_visual_path)
                except Exception as e:
                    print(f"Failed to load base visual: {e}")

        def make_frame(t):
            # 1. Base Canvas (Background Layer)
            frame = np.zeros((self.vh, self.vw, 3), dtype=np.uint8)
            idx = min(int((t * sr) / hop), len(pb)-1)
            
            s_idx = 0
            for i in range(len(markers)-1):
                if markers[i] <= idx < markers[i+1]: s_idx = i; break
            
            phase = section_phases[s_idx]
            engine, palette = engine_selector(t, idx, s_idx, phase)

            # --- Style or Visual Main Mode ---
            if base_clip is not None:
                try:
                    t_visual = t % base_clip.duration if base_clip.duration else 0
                    b_frame = base_clip.get_frame(t_visual)
                    b_frame = cv2.cvtColor(b_frame, cv2.COLOR_RGB2BGR)
                    
                    # Resize and crop to fill the canvas
                    bh, bw = b_frame.shape[:2]
                    target_ratio = self.vw / self.vh
                    current_ratio = bw / bh
                    
                    if current_ratio > target_ratio:
                        new_w = int(bh * target_ratio)
                        x_off = (bw - new_w) // 2
                        b_frame = b_frame[:, x_off:x_off+new_w]
                    else:
                        new_h = int(bw / target_ratio)
                        y_off = (bh - new_h) // 2
                        b_frame = b_frame[y_off:y_off+new_h, :]
                        
                    b_frame = cv2.resize(b_frame, (self.vw, self.vh))
                    
                    # Oto MAD effect 1: Intense Audio reactive zoom/bounce
                    zoom = 1.0 + 0.3 * pb[idx] + 0.1 * hm[idx]
                    if zoom > 1.0:
                        ch, cw = self.vh, self.vw
                        new_w, new_h = int(cw / zoom), int(ch / zoom)
                        x1, y1 = (cw - new_w) // 2, (ch - new_h) // 2
                        cropped = b_frame[y1:y1+new_h, x1:x1+new_w]
                        b_frame = cv2.resize(cropped, (cw, ch))

                    # Oto MAD effect 2: Beat flash (white flash on heavy beat)
                    if pb[idx] > 0.9 and (idx % 4 == 0):
                        flash_intensity = (pb[idx] - 0.9) * 10
                        b_frame = cv2.addWeighted(b_frame, 1.0, np.full_like(b_frame, 255), flash_intensity, 0)
                        
                    frame = b_frame
                except Exception as e:
                    engine.draw(frame, t, palette, hm[idx], pb[idx], hm[idx], sub_e[idx], mel_p[max(0, idx-100):idx], mel_e[max(0, idx-100):idx], "LR", phase, theme)
            else:
                # 2. Draw Style (Background/Midground)
                engine.draw(frame, t, palette, hm[idx], pb[idx], hm[idx], sub_e[idx], mel_p[max(0, idx-100):idx], mel_e[max(0, idx-100):idx], "LR", phase, theme)
            # -------------------------------
            
            # 3. Camera & Effects
            is_beat = pb[idx] > 0.8 and (idx % 8 == 0)
            camera.update(pb[idx], is_beat)
            frame = camera.apply(frame)
            frame = PostProcessor.apply_effects(frame, pb[idx], hm[idx], theme)
            
            # 4. Foreground Elements (Only for procedural styles)
            if base_clip is None:
                r = int(((80 + 20*phase) + 180 * pb[idx]) * theme.line_weight / 2)
                if ratio == "9:16": r = int(r * 0.7)
                
                if hasattr(engine, "draw_center"):
                    engine.draw_center(frame, r, palette, theme, phase, t)
                else:
                    cv2.circle(frame, (self.vw//2, self.vh//2), r, palette[2], int(theme.line_weight + phase), cv2.LINE_AA)
                
                cv2.circle(frame, (self.vw//2, self.vh//2), r + 20, (255, 255, 255), max(1, int((5 + 5*phase) * hm[idx])))
            
            # 5. Lyrics Layer (AI Generated)
            if lyrics_segments:
                current_text = ""
                for segment in lyrics_segments:
                    if segment['start'] <= t <= segment['end']:
                        current_text = segment['text']
                        break
                
                if current_text:
                    # Render text with PIL for Japanese support
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    
                    font_size = int(40 + 20 * pb[idx]) # Beat reactive size
                    try:
                        font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()
                    except:
                        font = ImageFont.load_default()
                    
                    # Calculate text size and position
                    bbox = draw.textbbox((0, 0), current_text, font=font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    text_x, text_y = (self.vw - text_w) // 2, int(self.vh * 0.85)
                    
                    # Draw shadow
                    draw.text((text_x+2, text_y+2), current_text, font=font, fill=(0,0,0))
                    # Draw main text (White or Palette color)
                    draw.text((text_x, text_y), current_text, font=font, fill=(255,255,255))
                    
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Transition flash
            if (idx - markers[s_idx]) < 10:
                flash = 1.0 - (idx - markers[s_idx]) / 10
                cv2.addWeighted(frame, 1.0, np.full_like(frame, 255), 0.4 * flash, 0, dst=frame)
            
            return frame

        return make_frame, duration
