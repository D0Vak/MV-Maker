import cv2
import numpy as np
from design_system import draw_vignette

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
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
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
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        for i in range(12):
            a = i * np.pi / 6
            cv2.line(frame, (int(cx + r*0.8*np.cos(a)), int(cy + r*0.8*np.sin(a))), 
                     (int(cx + r*1.5*np.cos(a)), int(cy + r*1.5*np.sin(a))), pal[2], int(theme.line_weight))
        cv2.circle(frame, (cx, cy), r, pal[1], -1)
        cv2.circle(frame, (cx, cy), int(r * 1.1), (255, 255, 255), 1)

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
        # 2. Particles with trails
        for i in range(int(50 * phase * theme.line_weight / 2)):
            np.random.seed(i * 123)
            px, py = (np.random.randint(0, w) + int(t*40))%w, (np.random.randint(0, h) + int(t*30))%h
            cv2.line(frame, (px, py), (px - 10, py - 5), pal[2], 1)
            cv2.circle(frame, (px, py), int(theme.line_weight), pal[2], -1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
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
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
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
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
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
        pass 

class StyleStormyLandscape:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        c1 = np.array(pal[0]) * 0.1
        c2 = np.array(pal[1]) * 0.2
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)
        for i in range(3):
            np.random.seed(i * 100)
            pts = [[0, h]]
            for x in range(0, w + 200, 200):
                y = h - (150 + i * 100) - np.random.randint(0, 100)
                pts.append([x, y])
            pts.append([w, h])
            cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c * (0.05 + i*0.1)) for c in pal[0]])
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
        if phase == 1: c1, c2 = np.array([20, 20, 40]), np.array([100, 60, 40])
        elif phase == 2: c1, c2 = np.array([150, 100, 40]), np.array([40, 40, 80])
        else: c1, c2 = np.array([10, 10, 20]), np.array([30, 20, 60])
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)
        for i in range(2):
            np.random.seed(i * 555)
            pts = [[0, h]]
            off = int(t * (10 + i*10)) % 400
            for x in range(-off, w + 400, 400):
                y = h - (100 + i*80) - np.random.randint(0, 50)
                pts.append([x, y])
            pts.append([w, h])
            cv2.fillPoly(frame, [np.array(pts, np.int32)], [int(c * (0.1 + i*0.1)) for c in pal[0]])
        ground_y = int(h * 0.8)
        cv2.rectangle(frame, (0, ground_y), (w, h), [int(c * 0.05) for c in pal[0]], -1)
        pole_spacing = 600
        pole_offset = int(t * 300) % pole_spacing
        for x in range(w + pole_spacing, -pole_spacing, -pole_spacing):
            px = x - pole_offset
            if px < -50 or px > w + 50: continue
            cv2.rectangle(frame, (px, ground_y - 400), (px + 15, ground_y), (5, 5, 10), -1)
            cv2.line(frame, (px, ground_y - 380), (px + pole_spacing, ground_y - 360), (10, 10, 15), 1)
        cel_y = int(h * 0.3 - 100 * np.sin(t * 0.1))
        color = (255, 255, 255) if phase == 3 else (200, 220, 255)
        r_cel = int(50 + 50 * pb)
        cv2.circle(frame, (w - 200, cel_y), r_cel, color, -1)
        if phase == 3: cv2.circle(frame, (w - 200, cel_y), r_cel + 30, color, 1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        pass 

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
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
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
            overlay = cv2.GaussianBlur(overlay, (0, 0), 20 + 10 * intensity)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        cv2.circle(frame, (cx, cy), int(r * 0.8), [int(c*0.5) for c in pal[1]], 2)
        cv2.circle(frame, (cx, cy), int(r * 0.4), pal[2], -1)

class StyleGrungeGrind:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.05) for c in pal[0]]
        for i in range(10 + 20 * phase):
            np.random.seed(i + int(t * 15))
            pts = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(3)], np.int32)
            cv2.polylines(frame, [pts], False, pal[1], int(theme.line_weight + 2 * intensity))
        if pb > 0.8:
            overlay = np.random.randint(0, 255, frame.shape, dtype=np.uint8)
            cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, dst=frame)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        for _ in range(5):
            pts = np.array([[cx + np.random.randint(-r, r), cy + np.random.randint(-r, r)] for _ in range(3)], np.int32)
            cv2.fillPoly(frame, [pts], pal[2])

class StylePopDynamic:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = pal[0]
        for i in range(5 + 5 * phase):
            np.random.seed(i * 44)
            bounce = abs(np.sin(t * 4 + i)) * 100 * intensity
            cx = int(w * (i+1) / 10)
            cy = int(h * 0.8 - bounce)
            cv2.circle(frame, (cx, cy), int(40 * (1 + intensity)), pal[1], -1)
            cv2.circle(frame, (cx, cy), int(50 * (1 + intensity)), pal[2], int(theme.line_weight))
        cv2.putText(frame, "POP_SYNC", (w-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, pal[2], 3)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        s = int(r * 1.2)
        cv2.rectangle(frame, (cx-s, cy-s), (cx+s, cy+s), pal[2], int(theme.line_weight * 2))
        cv2.putText(frame, "HIT!", (cx-20, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 1, pal[1], 4)

class StyleGraphicArt:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = [int(c * 0.95) for c in pal[0]] 
        for i in range(20 + 20 * phase):
            np.random.seed(i * 555)
            x, y = (np.random.randint(0, w) + int(t * 10)) % w, np.random.randint(0, h)
            cv2.circle(frame, (x, y), 3, [int(c*0.3) for c in pal[1]], -1)
        for i in range(2):
            np.random.seed(i * 123)
            r_big = 200 + 100 * intensity
            cv2.circle(frame, (w, 0), int(r_big), [int(c*0.2) for c in pal[2]], int(theme.line_weight * 5))
            cv2.circle(frame, (0, h), int(r_big * 0.5), [int(c*0.2) for c in pal[1]], -1)
        for i in range(10):
            bh = int(100 * pb * (1 + i*0.1))
            cv2.rectangle(frame, (100 + i*20, h-50-bh), (110 + i*20, h-50), pal[2], -1)
        cv2.putText(frame, "AUDIO_SYNC_V2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [int(c*0.3) for c in pal[0]], 2)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        for i in range(3):
            cv2.circle(frame, (cx, cy), r - i*20, pal[1], int(theme.line_weight))
        cv2.line(frame, (cx-r-20, cy), (cx+r+20, cy), pal[2], 2)
        cv2.line(frame, (cx, cy-r-20), (cx, cy+r+20), pal[2], 2)

class StyleMangaLayout:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = (240, 240, 240)
        panels = [(0, 0, w, h)]
        if intensity > 0.4: panels = [(0, 0, w//2 - 5, h), (w//2 + 5, 0, w, h)]
        if intensity > 0.7: panels = [(0, 0, w//2 - 5, h//2 - 5), (w//2 + 5, 0, w, h//2 - 5), (0, h//2 + 5, w, h)]
        for i, (x1, y1, x2, y2) in enumerate(panels):
            pw, ph = x2 - x1, y2 - y1
            if i == len(panels) - 1:
                cx, cy = x1 + pw//2, y1 + ph//2
                for a_idx in range(60):
                    angle = a_idx * np.pi / 30 + t * 0.5
                    length = 200 + 400 * intensity
                    cv2.line(frame, (cx, cy), (int(cx + length*np.cos(angle)), int(cy + length*np.sin(angle))), (20, 20, 20), 1)
            for _ in range(int(5 * intensity)):
                rx, ry = np.random.randint(x1, x2), np.random.randint(y1, y2)
                cv2.line(frame, (rx, ry), (rx + 50, ry - 20), (100, 100, 100), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        if pb > 0.85:
            text = np.random.choice(["DOGN!", "BANG!", "BOOM!", "ドドド"])
            cv2.putText(frame, text, (w//2 - 100, h//2), cv2.FONT_HERSHEY_TRIPLEX, 3 * intensity, (0, 0, 0), 8)

    def draw_center(self, frame, r, pal, theme, phase, t):
        pass

class StyleActionSilhouette:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        c1, c2 = np.array(pal[1]) * 0.2, np.array(pal[2]) * 0.4
        for y in range(h): frame[y, :] = (c1 * (1 - y/h) + c2 * (y/h)).astype(np.uint8)
        for i in range(20):
            np.random.seed(i)
            sx = (np.random.randint(0, w) - int(t * 1500)) % (w + 400) - 200
            sy = np.random.randint(0, h)
            slen = 200 + 300 * intensity
            cv2.line(frame, (int(sx), sy), (int(sx + slen), sy), (255, 255, 255), 1)
        cx, cy = w // 2, h // 2 + 100
        pose_seed = int(t * 8) % 4
        color = (10, 10, 10)
        cv2.circle(frame, (cx, cy - 200), 40, color, -1)
        cv2.line(frame, (cx, cy - 160), (cx, cy), color, 40)
        if pose_seed == 0:
            cv2.line(frame, (cx, cy), (cx - 60, cy + 100), color, 20)
            cv2.line(frame, (cx, cy), (cx + 80, cy + 40), color, 20)
        elif pose_seed == 1:
            cv2.line(frame, (cx, cy), (cx - 40, cy + 30), color, 20)
            cv2.line(frame, (cx, cy), (cx + 40, cy + 30), color, 20)
        else:
            cv2.line(frame, (cx, cy - 100), (cx + 150, cy - 150), color, 15)
            if pb > 0.8:
                cv2.line(frame, (cx - 100, cy - 200), (cx + 200, cy - 100), (255, 255, 255), 5)

    def draw_center(self, frame, r, pal, theme, phase, t):
        pass

class StyleClassic:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        c_bg = [int(c * 0.15) for c in pal[0]]
        frame[:, :] = c_bg
        border = 40
        c_gold = (100, 180, 210) 
        cv2.rectangle(frame, (border, border), (w-border, h-border), c_gold, 2)
        cv2.rectangle(frame, (border+10, border+10), (w-border-10, h-border-10), c_gold, 1)
        for i in range(10 + phase * 5):
            np.random.seed(i * 77)
            x = (np.random.randint(0, w) + int(t * 50)) % w
            y = np.random.randint(border, h - border)
            cv2.circle(frame, (x, y), 5, c_gold, -1)
            cv2.line(frame, (x+5, y), (x+5, y-30), c_gold, 2)
            if i % 3 == 0:
                cv2.line(frame, (x+5, y-30), (x+15, y-20), c_gold, 2)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        pts = np.array([[cx-20, cy+40], [cx+20, cy+40], [cx+40, cy-20], [cx, cy-40], [cx-40, cy-20]], np.int32)
        cv2.polylines(frame, [pts], True, (100, 180, 210), 2)
        cv2.circle(frame, (cx, cy), r, (100, 180, 210), 1)

class StyleDarkFantasy:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = (5, 0, 10) 
        cx, cy = w//2, h//2
        r_base = int(200 + 100 * np.sin(t * 0.5))
        cv2.circle(frame, (cx, cy), r_base, (40, 20, 80), 2, cv2.LINE_AA)
        for i in range(3):
            a = t + i * 2 * np.pi / 3
            cv2.line(frame, (cx, cy), (int(cx + r_base * np.cos(a)), int(cy + r_base * np.sin(a))), (60, 30, 100), 1)
        for i in range(40 + phase * 20):
            np.random.seed(i * 13)
            px = np.random.randint(0, w)
            py = (np.random.randint(0, h) - int(t * 100)) % h
            size = np.random.randint(1, 4)
            c_ember = (20, 20, 200) if np.random.random() > 0.5 else (80, 40, 150)
            cv2.circle(frame, (px, py), size, c_ember, -1)
        if pb > 0.8:
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), int(r_base * 1.5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3 * intensity, frame, 0.7, 0, dst=frame)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        for i in range(8):
            a = i * np.pi / 4 + t
            cv2.putText(frame, "V", (int(cx + r * np.cos(a)), int(cy + r * np.sin(a))), 
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (100, 50, 255), 1)
        cv2.circle(frame, (cx, cy), r, (150, 50, 200), 2)

class StyleCyberPunk:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = (20, 10, 10)
        # 1. Digital Grid
        for i in range(0, w, 40): cv2.line(frame, (i, 0), (i, h), (40, 20, 40), 1)
        for i in range(0, h, 40): cv2.line(frame, (0, i), (w, i), (40, 20, 40), 1)
        # 2. Scanning Bars
        sy = int(t * 200) % h
        cv2.line(frame, (0, sy), (w, sy), (100, 255, 255), 2)
        # 3. Random Tech Boxes
        for i in range(int(5 * intensity)):
            np.random.seed(i + int(t * 5))
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 0, 255), 1)
            cv2.putText(frame, "SYSTEM_LINK", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        cv2.rectangle(frame, (cx-r, cy-r), (cx+r, cy+r), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), int(r*0.5), (255, 0, 255), -1)

class StyleLofiChill:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = (180, 160, 160) # Warm sepia
        # 1. Dust & Scratches
        for _ in range(int(10 * intensity)):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            cv2.circle(frame, (x, y), 1, (100, 100, 100), -1)
        # 2. Spinning Record Shadow
        cx, cy = w-200, h-200
        cv2.circle(frame, (cx, cy), 150, (50, 50, 50), -1)
        cv2.circle(frame, (cx, cy), 155, (200, 180, 180), 2)
        # 3. Soft Floating Circles
        for i in range(5):
            np.random.seed(i)
            ox, oy = int(100 * np.cos(t*0.5 + i)), int(100 * np.sin(t*0.4 + i))
            cv2.circle(frame, (w//2 + ox, h//2 + oy), 100, (220, 200, 200), -1)

    def draw_center(self, frame, r, pal, theme, phase, t):
        pass

class StyleHeavyMetal:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        frame[:, :] = (10, 10, 10)
        # 1. Jagged Shards
        for i in range(10 + int(20 * pb)):
            np.random.seed(i + int(t * 10))
            pts = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(3)], np.int32)
            cv2.fillPoly(frame, [pts], (40, 40, 120) if np.random.random() > 0.5 else (20, 20, 60))
        # 2. Fire Sparks
        for i in range(50):
            np.random.seed(i * 12)
            sx = np.random.randint(0, w)
            sy = (np.random.randint(0, h) - int(t * 800)) % h
            cv2.line(frame, (sx, sy), (sx, sy+20), (50, 100, 255), 2) # Blue fire sparks
        # 3. Visual Shake is handled by Camera, but we add static noise
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        cv2.add(frame, noise, dst=frame)

    def draw_center(self, frame, r, pal, theme, phase, t):
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        cv2.polylines(frame, [np.array([[cx-r, cy], [cx, cy-r], [cx+r, cy], [cx, cy+r]], np.int32)], True, (100, 150, 255), 4)

class StyleSynthWave:
    def draw(self, frame, t, pal, intensity, pb, hm, se, mel_p, mel_e, mode, phase, theme):
        h, w = frame.shape[:2]
        # 1. Gradient Sunset
        for y in range(h):
            c = (255 * (y/h), 0, 255 * (1 - y/h)) # Pink to Blue
            frame[y, :] = [int(x * 0.3) for x in c]
        # 2. Perspected Grid
        cx, cy = w//2, h//2
        for i in range(-10, 11):
            cv2.line(frame, (cx + i * 100, cy), (cx + i * 1000, h), (255, 0, 255), 1)
        for i in range(10):
            gy = cy + int(pow(i/10, 2) * (h - cy))
            cv2.line(frame, (0, gy), (w, gy), (255, 0, 255), 1)
        # 3. Big Sun
        sun_r = 150 + int(20 * pb)
        cv2.circle(frame, (cx, cy - 50), sun_r, (0, 100, 255), -1)
        for i in range(5):
            sy = cy - 50 + 40 * i
            cv2.line(frame, (cx-sun_r, sy), (cx+sun_r, sy), (20, 10, 10), 4)

    def draw_center(self, frame, r, pal, theme, phase, t):
        pass
