"""
main_inference.py  —  Redesign UI (3-Panel Dashboard) + Arduino LED Control
===========================================================================
Layout layar penuh dengan 3 panel + kontrol LED via Arduino Nano.

  LED Mapping:
    D9  = Lirik Kiri  (KUNING)
    D10 = Center      (HIJAU)  
    D11 = Lirik Kanan (BIRU)
    
  Protokol Serial: 'L'=Kiri, 'C'=Center, 'R'=Kanan, 'N'=Off
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import math
import serial
import serial.tools.list_ports
import glob
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════════════════════════════
IMG_SIZE             = 64
MODEL_PATH           = "model/eye_tracker.h5"
LABELS               = ["Lirik Kiri", "Lirik Kanan", "Center"]
LABEL_COLORS         = {                       # BGR
    "Lirik Kiri" : (255, 210,   0),
    "Lirik Kanan": ( 30, 130, 255),
    "Center"     : ( 80, 255, 120),
}
CONFIDENCE_THRESHOLD = 0.60
SMOOTH_WINDOW        = 7

# ─── ARDUINO CONFIG ───────────────────────────────────────────────────────────
ARDUINO_BAUDRATE     = 9600
ARDUINO_CMD_MAP      = {
    "Lirik Kiri" : b'L',   # D9
    "Center"     : b'C',   # D10
    "Lirik Kanan": b'R',   # D11
}
ARDUINO_CMD_OFF      = b'N'

# Dimensi Panel
PANEL_LEFT_W  = 200
PANEL_RIGHT_W = 185
TOPBAR_H      = 44
BOTTOMBAR_H   = 34

# Warna tema (BGR) — dark theme
BG_DARK    = (14,  14,  12)
BG_PANEL   = (18,  18,  20)
BG_CARD    = (26,  26,  30)
BG_VIDEO   = (10,  10,  12)
LINE_DIM   = (50,  50,  55)
TEXT_PRI   = (220, 218, 210)
TEXT_SEC   = (130, 128, 120)
TEXT_DIM   = ( 80,  80,  85)
ACCENT_GRN = ( 80, 222,  74)
ACCENT_YEL = ( 50, 158, 245)
ACCENT_GRY = (100, 100, 105)
ACCENT_RED = ( 60,  60, 255)  # Arduino disconnect

# ═══════════════════════════════════════════════════════════════════════════════
# ARDUINO HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def find_arduino_port():
    """Auto-detect Arduino Nano port across platforms."""
    # Coba port umum berdasarkan OS
    if sys.platform.startswith('linux'):
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    elif sys.platform == 'darwin':  # macOS
        ports = glob.glob('/dev/tty.usbserial*') + glob.glob('/dev/tty.wchusbserial*')
    elif sys.platform == 'win32':   # Windows
        ports = [f'COM{i}' for i in range(1, 20)]
    else:
        ports = []
    
    # Cek port yang tersedia
    available = [p.device for p in serial.tools.list_ports.comports()]
    
    for p in ports:
        if p in available:
            try:
                s = serial.Serial(p, ARDUINO_BAUDRATE, timeout=1)
                s.close()
                return p
            except (OSError, serial.SerialException):
                continue
    
    # Fallback: scan semua port yang terdeteksi
    for p in available:
        # Filter berdasarkan nama hardware
        desc = next((d.description for d in serial.tools.list_ports.comports() 
                     if d.device == p), "")
        if any(k in desc.lower() for k in ['arduino', 'ch340', 'ft232', 'usb-serial']):
            try:
                s = serial.Serial(p, ARDUINO_BAUDRATE, timeout=1)
                s.close()
                return p
            except (OSError, serial.SerialException):
                continue
    
    return None

def init_arduino():
    """Inisialisasi koneksi ke Arduino Nano."""
    port = find_arduino_port()
    if port is None:
        print("⚠️  Arduino tidak terdeteksi. LED tidak akan aktif.")
        print("   Tips: Hubungkan Arduino via USB dan pastikan driver CH340 terinstall.")
        return None
    
    try:
        ard = serial.Serial(port, ARDUINO_BAUDRATE, timeout=1)
        time.sleep(2)  # Tunggu Arduino reset selesai
        print(f"✅  Arduino terhubung di {port}")
        print(f"    Baudrate: {ARDUINO_BAUDRATE}")
        # Kirim 'N' untuk memastikan LED mati saat mulai
        ard.write(ARDUINO_CMD_OFF)
        return ard
    except serial.SerialException as e:
        print(f"⚠️  Gagal membuka port {port}: {e}")
        return None

def send_to_arduino(ard, prediction_label):
    """Kirim perintah ke Arduino berdasarkan prediksi."""
    if ard is None or not ard.is_open:
        return False
    
    cmd = ARDUINO_CMD_MAP.get(prediction_label, ARDUINO_CMD_OFF)
    try:
        ard.write(cmd)
        return True
    except serial.SerialException:
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("Memuat model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅  Model: {MODEL_PATH}")
print(f"    Input : {model.input_shape}")
print(f"    Output: {model.output_shape}")

# ─── HAAR CASCADE ──────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_eye(eye_roi):
    resized   = cv2.resize(eye_roi, (IMG_SIZE, IMG_SIZE))
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(resized)
    thresh    = cv2.adaptiveThreshold(
        equalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=11, C=2)
    return thresh.astype(np.float32) / 255.0

def filter_eyes(eyes, roi_height, roi_width):
    valid    = []
    min_size = int(roi_width * 0.12)
    max_size = int(roi_width * 0.55)
    for (ex, ey, ew, eh) in eyes:
        if ey > roi_height * 0.65:
            continue
        if ew < min_size or ew > max_size:
            continue
        valid.append((ex, ey, ew, eh))
    valid = sorted(valid, key=lambda e: e[2]*e[3], reverse=True)
    return valid[:2]

def predict_eyes(eye_frames):
    if len(eye_frames) == 0:
        return np.ones(3) / 3
    all_probs = []
    for ef in eye_frames:
        inp   = ef[np.newaxis, ..., np.newaxis]
        probs = model.predict(inp, verbose=0)[0]
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════════
pred_history = []

def smooth_prediction(probs):
    pred_history.append(probs)
    if len(pred_history) > SMOOTH_WINDOW:
        pred_history.pop(0)
    return np.mean(pred_history, axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER DRAWING
# ═══════════════════════════════════════════════════════════════════════════════
def filled_rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, -1)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        img[:] = overlay

def border_rounded_rect(img, x, y, w, h, r, color, thickness=1):
    cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, thickness)
    cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, thickness)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, color, thickness)

def put_text(img, text, x, y, scale=0.42, color=TEXT_PRI, bold=False):
    thickness = 2 if bold else 1
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def section_label(img, text, x, y):
    put_text(img, text.upper(), x, y, scale=0.30, color=TEXT_DIM)

def draw_arc(img, cx, cy, radius, start_deg, end_deg, color, thickness=6):
    cv2.ellipse(img, (cx, cy), (radius, radius),
                0, start_deg, end_deg, color, thickness, cv2.LINE_AA)

def draw_divider(img, x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2, y2), LINE_DIM, 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_topbar(frame, W, fps, arduino_connected):
    bar = frame[:TOPBAR_H, :]
    bar[:] = (22, 22, 24)
    draw_divider(frame, 0, TOPBAR_H-1, W, TOPBAR_H-1)

    # Live dot
    dot_color = ACCENT_GRN
    cv2.circle(bar, (16, TOPBAR_H//2), 5, dot_color, -1, cv2.LINE_AA)
    cv2.circle(bar, (16, TOPBAR_H//2), 5, (255,255,255), 1, cv2.LINE_AA)

    put_text(bar, "Eye Tracker", 28, TOPBAR_H//2 + 5, scale=0.50,
             color=TEXT_PRI, bold=True)

    # ── Arduino Status Indicator ─────────────────────────────────────────────
    ard_x = W - 10
    if arduino_connected:
        ard_text = "Arduino ON"
        ard_color = ACCENT_GRN
        ard_bg = (20, 60, 20)
    else:
        ard_text = "Arduino OFF"
        ard_color = ACCENT_RED
        ard_bg = (40, 20, 20)
    
    (tw, _), _ = cv2.getTextSize(ard_text, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    ard_x -= tw + 8
    filled_rounded_rect(bar, ard_x-4, 12, tw+12, 20, 3, ard_bg)
    put_text(bar, ard_text, ard_x, TOPBAR_H//2+5, scale=0.33, color=ard_color)
    ard_x -= 18  # space before pipe

    # FPS
    fps_text = f"FPS: {fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    ard_x -= tw + 8
    fps_box_x = ard_x - 4
    filled_rounded_rect(bar, fps_box_x, 12, tw+12, 20, 3, (20, 60, 20))
    border_rounded_rect(bar, fps_box_x, 12, tw+12, 20, 3, (40, 140, 40), 1)
    put_text(bar, fps_text, ard_x, TOPBAR_H//2+5, scale=0.40, color=ACCENT_GRN)

    infos = [
        f"Model: eye_tracker.h5",
        f"Input: 64x64",
        f"Window: {SMOOTH_WINDOW}f",
    ]
    ard_x -= 10
    for info in reversed(infos):
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        ard_x -= tw + 20
        put_text(bar, info, ard_x, TOPBAR_H//2+5, scale=0.33, color=TEXT_SEC)
        put_text(bar, "|", ard_x - 10, TOPBAR_H//2+5, scale=0.33, color=TEXT_DIM)


def build_bottombar(frame, H, W, arduino_port):
    by = H - BOTTOMBAR_H
    bar = frame[by:H, :]
    bar[:] = (22, 22, 24)
    draw_divider(frame, 0, by, W, by)

    port_text = f"Arduino: {arduino_port}" if arduino_port else "Arduino: Not connected"
    items = [
        ("\u25b6 Kamera: /dev/video0", TEXT_SEC),
        ("|", TEXT_DIM),
        ("TF 2.x", TEXT_SEC),
        ("|", TEXT_DIM),
        ("Haar Cascade", TEXT_SEC),
        ("|", TEXT_DIM),
        (port_text, ACCENT_GRN if arduino_port else ACCENT_RED),
    ]
    x = 12
    for txt, col in items:
        put_text(bar, txt, x, BOTTOMBAR_H//2 + 5, scale=0.33, color=col)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        x += tw + 8

    shortcuts = [("[Q] Keluar", W-130), ("[S] Screenshot", W-230)]
    for txt, sx in shortcuts:
        filled_rounded_rect(bar, sx-2, 8, 90, 18, 3, (36,36,40))
        border_rounded_rect(bar, sx-2, 8, 90, 18, 3, (70,70,75), 1)
        put_text(bar, txt, sx+2, BOTTOMBAR_H//2+5, scale=0.33, color=TEXT_SEC)


def build_left_panel(frame, H, W,
                     eye_count, eye_previews,
                     prediction_label, prediction_color,
                     conf_threshold, conf_threshold_pct):
    panel = frame[TOPBAR_H:H-BOTTOMBAR_H, 0:PANEL_LEFT_W]
    panel[:] = BG_PANEL
    draw_divider(frame, PANEL_LEFT_W, TOPBAR_H, PANEL_LEFT_W, H-BOTTOMBAR_H)

    PW = PANEL_LEFT_W
    px = 12
    py = 14

    # ── EYE PREVIEW ────────────────────────────────────────────────────────────
    section_label(panel, "Eye Preview", px, py + 8)
    py += 18

    box_w = (PW - px*2 - 8) // 2
    box_h = box_w + 20

    for i, (side, preview) in enumerate([("Kiri", eye_previews[0]),
                                          ("Kanan", eye_previews[1])]):
        bx = px + i * (box_w + 8)
        filled_rounded_rect(panel, bx, py, box_w, box_h, 6, BG_CARD)
        border_rounded_rect(panel, bx, py, box_w, box_h, 6,
                            (50,50,55), 1)
        if preview is not None:
            thumb = cv2.cvtColor(
                (preview * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            thumb = cv2.resize(thumb, (box_w-12, box_w-12))
            iy = py + 4
            panel[iy:iy+thumb.shape[0], bx+6:bx+6+thumb.shape[1]] = thumb
        else:
            cx = bx + box_w // 2
            cy = py + (box_w - 12) // 2 + 4
            pts = np.array([[bx+8, cy], [cx, py+10],
                            [bx+box_w-8, cy], [cx, py+box_w-10]], np.int32)
            cv2.polylines(panel, [pts], True, (50,50,55), 1, cv2.LINE_AA)

        put_text(panel, side, bx + box_w//2 - 12, py+box_h-4,
                 scale=0.33, color=TEXT_SEC)

    py += box_h + 14

    # ── DETEKSI ─────────────────────────────────────────────────────────────────
    section_label(panel, "Deteksi", px, py)
    py += 14

    filled_rounded_rect(panel, px, py, PW-px*2, 54, 6, BG_CARD)
    border_rounded_rect(panel, px, py, PW-px*2, 54, 6, (50,50,55), 1)

    ec_color = ACCENT_GRN if eye_count==2 else (ACCENT_YEL if eye_count==1 else ACCENT_GRY)
    put_text(panel, "Mata terdeteksi", px+8, py+18, scale=0.36, color=TEXT_SEC)
    put_text(panel, f"{eye_count} / 2", PW-px-35, py+18,
             scale=0.44, color=ec_color, bold=True)

    # progress bar
    bar_x, bar_y = px+8, py+30
    bar_w = PW - px*2 - 16
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+6),
                  (38,38,42), -1)
    filled = int(bar_w * eye_count / 2)
    if filled > 0:
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+filled, bar_y+6),
                      ec_color, -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+6),
                  (55,55,60), 1)

    hint = ("Averaging 2 mata" if eye_count==2
            else "1 mata terdeteksi" if eye_count==1
            else "Tidak ada mata")
    put_text(panel, hint, px+8, py+50, scale=0.33, color=ec_color)

    py += 68

    # ── STATUS PILL ─────────────────────────────────────────────────────────────
    section_label(panel, "Status", px, py)
    py += 14

    pill_color_bg = (
        (10, 40, 10) if eye_count==2 else
        (10, 30, 40) if eye_count==1 else
        (28, 28, 32)
    )
    pill_color_bd = ec_color
    filled_rounded_rect(panel, px, py, PW-px*2, 28, 6, pill_color_bg)
    border_rounded_rect(panel, px, py, PW-px*2, 28, 6,
                        tuple(c//3 for c in ec_color), 1)
    cv2.circle(panel, (px+14, py+14), 4, ec_color, -1, cv2.LINE_AA)

    status_txt = (
        prediction_label if eye_count > 0 and prediction_label not in
        ("Mencari wajah...", "Ragu") else
        "Mencari wajah..." if eye_count == 0 else prediction_label
    )
    put_text(panel, status_txt, px+24, py+19, scale=0.36, color=ec_color)

    py += 42

    # ── THRESHOLD SLIDER (visual) ────────────────────────────────────────────────
    section_label(panel, "Confidence Threshold", px, py)
    py += 14

    filled_rounded_rect(panel, px, py, PW-px*2, 38, 6, BG_CARD)
    border_rounded_rect(panel, px, py, PW-px*2, 38, 6, (50,50,55), 1)
    put_text(panel, "Min. threshold", px+8, py+16, scale=0.33, color=TEXT_SEC)
    put_text(panel, f"{int(conf_threshold_pct)}%", PW-px-28, py+16,
             scale=0.40, color=TEXT_PRI, bold=True)

    sl_x, sl_y = px+8, py+26
    sl_w = PW - px*2 - 16
    cv2.rectangle(panel, (sl_x, sl_y), (sl_x+sl_w, sl_y+4), (38,38,42), -1)
    pos = int(sl_w * conf_threshold_pct / 100)
    cv2.rectangle(panel, (sl_x, sl_y), (sl_x+pos, sl_y+4), ACCENT_GRN, -1)
    cv2.rectangle(panel, (sl_x, sl_y), (sl_x+sl_w, sl_y+4), (55,55,60), 1)
    cv2.circle(panel, (sl_x+pos, sl_y+2), 6, ACCENT_GRN, -1, cv2.LINE_AA)
    cv2.circle(panel, (sl_x+pos, sl_y+2), 6, TEXT_PRI, 1, cv2.LINE_AA)

    # ── LED STATUS ─────────────────────────────────────────────────────────────
    py += 50
    section_label(panel, "LED Output", px, py)
    py += 14

    filled_rounded_rect(panel, px, py, PW-px*2, 60, 6, BG_CARD)
    border_rounded_rect(panel, px, py, PW-px*2, 60, 6, (50,50,55), 1)

    led_labels = [("D9 Kiri", 0), ("D10 Mid", 1), ("D11 Kanan", 2)]
    active_map = {"Lirik Kiri": 0, "Center": 1, "Lirik Kanan": 2}

    for i, (lbl, idx) in enumerate(led_labels):
        active = eye_count > 0 and active_map.get(prediction_label) == idx
        lx = px + 8 + i * (PW - px*2 - 16) // 3
        led_color = (0, 255, 0) if active else (40, 40, 45)
        cv2.circle(panel, (lx + 6, py + 16), 5, led_color, -1, cv2.LINE_AA)
        if active:
            cv2.circle(panel, (lx + 6, py + 16), 5, (255,255,255), 1, cv2.LINE_AA)
        put_text(panel, lbl, lx - 2, py + 36, scale=0.28,
                color=ACCENT_GRN if active else TEXT_DIM)


def build_right_panel(frame, H, W, probs, smoothed_probs,
                      confidence, eye_count, prediction_label):
    rx0 = W - PANEL_RIGHT_W
    panel = frame[TOPBAR_H:H-BOTTOMBAR_H, rx0:W]
    panel[:] = BG_PANEL
    draw_divider(frame, rx0, TOPBAR_H, rx0, H-BOTTOMBAR_H)

    PW = PANEL_RIGHT_W
    px = 12
    py = 14

    # ── PROB BARS ───────────────────────────────────────────────────────────────
    section_label(panel, "Probabilitas", px, py)
    py += 14

    label_colors_bgr = {
        "Lirik Kiri" : (255, 210,   0),
        "Lirik Kanan": ( 30, 130, 255),
        "Center"     : ( 80, 255, 120),
    }
    pred_idx = int(np.argmax(smoothed_probs))

    for i, (label, prob) in enumerate(zip(LABELS, smoothed_probs)):
        is_top = (i == pred_idx)
        col = label_colors_bgr.get(label, TEXT_SEC)
        dim_col = tuple(max(c//4, 10) for c in col)

        filled_rounded_rect(panel, px, py, PW-px*2, 42, 5, BG_CARD)
        if is_top and eye_count > 0:
            border_rounded_rect(panel, px, py, PW-px*2, 42, 5,
                                tuple(c//2 for c in col), 1)
        else:
            border_rounded_rect(panel, px, py, PW-px*2, 42, 5, (50,50,55), 1)

        short = label.replace("Lirik ", "")
        name_col = col if is_top and eye_count>0 else TEXT_SEC
        put_text(panel, short, px+8, py+16, scale=0.36, color=name_col)
        pct_text = f"{prob*100:.0f}%"
        (tw, _), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        pct_col = col if is_top and eye_count>0 else TEXT_DIM
        put_text(panel, pct_text, PW-px-tw-4, py+16,
                 scale=0.40, color=pct_col, bold=is_top)

        bar_x, bar_y = px+8, py+26
        bar_w = PW - px*2 - 16
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+6),
                      (38,38,42), -1)
        filled = int(bar_w * prob)
        if filled > 0:
            bar_col = col if eye_count > 0 else ACCENT_GRY
            cv2.rectangle(panel, (bar_x, bar_y),
                          (bar_x+filled, bar_y+6), bar_col, -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+6),
                      (55,55,60), 1)

        py += 52

    py += 4

    # ── SMOOTHING INFO ────────────────────────────────────────────────────────
    section_label(panel, "Smoothing", px, py)
    py += 14

    filled_rounded_rect(panel, px, py, PW-px*2, 52, 6, BG_CARD)
    border_rounded_rect(panel, px, py, PW-px*2, 52, 6, (50,50,55), 1)

    half_w = (PW - px*2) // 2

    put_text(panel, f"{len(pred_history)}", px + half_w//2 - 8, py+30,
             scale=0.70, color=TEXT_PRI, bold=True)
    put_text(panel, "buffered", px+4, py+46, scale=0.30, color=TEXT_DIM)

    buf_color = ACCENT_GRN if len(pred_history)==SMOOTH_WINDOW else ACCENT_YEL
    put_text(panel, f"{SMOOTH_WINDOW}", px + half_w + half_w//2 - 8, py+30,
             scale=0.70, color=buf_color, bold=True)
    put_text(panel, "window", px + half_w + 4, py+46, scale=0.30, color=TEXT_DIM)
    cv2.line(panel, (px + half_w, py+8), (px + half_w, py+46), (55,55,60), 1)

    py += 62

    # ── CONFIDENCE ARC GAUGE ──────────────────────────────────────────────────
    section_label(panel, "Confidence", px, py)
    py += 14

    filled_rounded_rect(panel, px, py, PW-px*2, 90, 6, BG_CARD)
    border_rounded_rect(panel, px, py, PW-px*2, 90, 6, (50,50,55), 1)

    cx = PW // 2
    cy = py + 58
    r  = 36

    draw_arc(frame[TOPBAR_H:H-BOTTOMBAR_H, rx0:W],
             cx, cy, r, 150, 390, (40,40,44), 7)

    pred_col = label_colors_bgr.get(prediction_label, ACCENT_GRY)
    if eye_count == 0:
        pred_col = ACCENT_GRY
    arc_end = 150 + int(240 * confidence)
    if arc_end > 150:
        draw_arc(frame[TOPBAR_H:H-BOTTOMBAR_H, rx0:W],
                 cx, cy, r, 150, arc_end, pred_col, 7)

    tick_angle = 150 + int(240 * CONFIDENCE_THRESHOLD)
    tick_rad   = math.radians(tick_angle)
    tx1 = int(cx + (r-10) * math.cos(tick_rad))
    ty1 = int(cy + (r-10) * math.sin(tick_rad))
    tx2 = int(cx + (r+10) * math.cos(tick_rad))
    ty2 = int(cy + (r+10) * math.sin(tick_rad))
    cv2.line(panel, (tx1, ty1), (tx2, ty2), (200,200,200), 1, cv2.LINE_AA)

    put_text(panel, f"{confidence*100:.0f}%", cx-16, cy+6,
             scale=0.55, color=TEXT_PRI, bold=True)
    put_text(panel, prediction_label if eye_count>0 else "---",
             max(px+2, cx - len(prediction_label)*4), cy+22,
             scale=0.30, color=pred_col if eye_count>0 else TEXT_DIM)

    py += 100

    # ── AVERAGING INFO ───────────────────────────────────────────────────────
    section_label(panel, "Metode", px, py)
    py += 14

    filled_rounded_rect(panel, px, py, PW-px*2, 52, 6, BG_CARD)
    border_rounded_rect(panel, px, py, PW-px*2, 52, 6, (50,50,55), 1)

    lines = [
        "Prediksi dirata-rata",
        "dari semua mata valid.",
        "2 mata = lebih akurat.",
    ]
    for li, ln in enumerate(lines):
        put_text(panel, ln, px+8, py+14+li*14, scale=0.30, color=TEXT_DIM)


def letterbox(cam_frame, target_w, target_h, bg_color=BG_VIDEO):
    src_h, src_w = cam_frame.shape[:2]
    scale  = min(target_w / src_w, target_h / src_h)
    new_w  = int(src_w * scale)
    new_h  = int(src_h * scale)
    pad_x  = (target_w - new_w) // 2
    pad_y  = (target_h - new_h) // 2

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    resized = cv2.resize(cam_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas, scale, pad_x, pad_y


def build_video_area(frame, cam_frame, H, W,
                     faces, roi_data, prediction_label, prediction_color):
    vx0 = PANEL_LEFT_W
    vx1 = W - PANEL_RIGHT_W
    vy0 = TOPBAR_H
    vy1 = H - BOTTOMBAR_H
    vw  = vx1 - vx0
    vh  = vy1 - vy0

    video_area = frame[vy0:vy1, vx0:vx1]

    canvas, scale, pad_x, pad_y = letterbox(cam_frame, vw, vh)
    video_area[:] = canvas

    cam_h, cam_w = cam_frame.shape[:2]

    def cam2vid(cx, cy):
        return int(cx * scale + pad_x), int(cy * scale + pad_y)

    for (fx, fy, fw, fh) in faces[:1]:
        sx, sy = cam2vid(fx, fy)
        ex2, ey2 = cam2vid(fx+fw, fy+fh)
        sw, sh = ex2-sx, ey2-sy
        border_rounded_rect(video_area, sx, sy, sw, sh, 6, (50, 100, 180), 1)
        put_text(video_area, "wajah", sx+4, max(sy-4, 10),
                 scale=0.30, color=(100, 140, 200))

    for (ex_abs, ey_abs, ew, eh, roi_h, roi_w, _cam_h, _cam_w) in roi_data:
        sx, sy   = cam2vid(ex_abs, ey_abs)
        ex2, ey2 = cam2vid(ex_abs+ew, ey_abs+eh)
        sw, sh   = ex2-sx, ey2-sy
        border_rounded_rect(video_area, sx, sy, sw, sh, 4, (50, 230, 90), 1)
        cx = sx + sw // 2
        cy = sy + sh // 2
        cv2.circle(video_area, (cx, cy), 4, (50, 230, 90), -1, cv2.LINE_AA)
        cv2.circle(video_area, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_prediction_chip(frame, H, W, prediction_label, prediction_color, confidence):
    vx0 = PANEL_LEFT_W
    vx1 = W - PANEL_RIGHT_W
    vy0 = TOPBAR_H

    chip_text = prediction_label
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.70
    thick = 2
    (tw, th), _ = cv2.getTextSize(chip_text, font, scale, thick)

    chip_w = tw + 32
    chip_h = th + 22
    chip_x = vx0 + (vx1 - vx0)//2 - chip_w//2
    chip_y = vy0 + 12

    overlay = frame.copy()
    filled_rounded_rect(overlay, chip_x, chip_y, chip_w, chip_h, 8,
                        (20, 20, 22))
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    border_rounded_rect(frame, chip_x, chip_y, chip_w, chip_h, 8,
                        tuple(c//2 for c in prediction_color), 1)

    tx = chip_x + 16
    ty = chip_y + chip_h - 10
    cv2.putText(frame, chip_text, (tx, ty), font, scale,
                prediction_color, thick, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("   Eye Tracker + Arduino LED Controller")
print("=" * 60)

# ── Inisialisasi Arduino ─────────────────────────────────────────────────────
arduino = init_arduino()
arduino_port = arduino.port if arduino else None
arduino_connected = arduino is not None

# ── Inisialisasi Kamera ──────────────────────────────────────────────────────
cap       = cv2.VideoCapture(0)
prev_time = time.time()

_, init_frame = cap.read()
if init_frame is None:
    print("ERROR: Kamera tidak terdeteksi.")
    if arduino:
        arduino.close()
    exit()

WIN_NAME = "Eye Tracker — Dashboard + Arduino"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)

print("\nEye Tracker aktif. Tekan Q untuk keluar.")
print("Tips: Pastikan wajah terkena cahaya dari depan.")

conf_threshold_pct = CONFIDENCE_THRESHOLD * 100

# Variabel untuk tracking perubahan prediksi (hanya kirim saat berubah)
last_sent_label = None

while True:
    ret, cam_frame = cap.read()
    if not ret:
        break

    # Tentukan ukuran canvas dari ukuran window saat ini
    win_rect = cv2.getWindowImageRect(WIN_NAME)
    W = max(win_rect[2], 800) if win_rect[2] > 0 else 1280
    H = max(win_rect[3], 500) if win_rect[3] > 0 else 720

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = BG_DARK

    # Deteksi wajah & mata pada frame ASLI
    gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    prediction_label = "Mencari wajah..."
    prediction_color = ACCENT_GRY
    current_probs    = np.ones(3) / 3
    eye_count        = 0
    eye_previews     = [None, None]
    roi_overlays     = []

    cam_h, cam_w = cam_frame.shape[:2]

    for (fx, fy, fw, fh) in faces[:1]:
        roi_h    = int(fh * 0.60)
        roi_gray = gray[fy:fy+roi_h, fx:fx+fw]

        raw_eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=6, minSize=(25, 25))

        filtered_eyes = filter_eyes(raw_eyes, roi_h, fw) if len(raw_eyes) > 0 else []
        eye_count     = len(filtered_eyes)
        eye_frames    = []

        for idx, (ex, ey, ew, eh) in enumerate(filtered_eyes):
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            processed  = preprocess_eye(eye_region)
            eye_frames.append(processed)
            if idx < 2:
                eye_previews[idx] = processed
            roi_overlays.append((fx+ex, fy+ey, ew, eh, roi_h, fw, cam_h, cam_w))

        if len(eye_frames) > 0:
            raw_probs      = predict_eyes(eye_frames)
            smoothed_probs = smooth_prediction(raw_probs)
            current_probs  = smoothed_probs

            pred_idx   = int(np.argmax(smoothed_probs))
            confidence = float(smoothed_probs[pred_idx])

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_label = LABELS[pred_idx]
                prediction_color = LABEL_COLORS[prediction_label]
            else:
                prediction_label = f"Ragu ({confidence*100:.0f}%)"
                prediction_color = ACCENT_GRY
        break

    confidence = float(np.max(current_probs))

    # ── KIRIM KE ARDUINO (hanya saat prediksi berubah) ────────────────────────
    if eye_count > 0 and prediction_label in ARDUINO_CMD_MAP:
        if prediction_label != last_sent_label:
            success = send_to_arduino(arduino, prediction_label)
            if success:
                last_sent_label = prediction_label
    elif eye_count == 0:
        if last_sent_label is not None:
            send_to_arduino(arduino, None)  # Kirim 'N' (off)
            last_sent_label = None

    # Cek koneksi Arduino
    arduino_connected = arduino is not None and arduino.is_open

    # Mirror untuk display
    cam_display = cv2.flip(cam_frame, 1)

    faces_flipped = []
    for (fx, fy, fw, fh) in faces[:1]:
        faces_flipped.append((cam_w - fx - fw, fy, fw, fh))

    roi_flipped = []
    for (ex_abs, ey_abs, ew, eh, roi_h, roi_w, _ch, _cw) in roi_overlays:
        roi_flipped.append((cam_w - ex_abs - ew, ey_abs, ew, eh,
                            roi_h, roi_w, cam_h, cam_w))

    # FPS
    curr_time = time.time()
    fps       = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    # Bangun UI
    build_topbar(frame, W, fps, arduino_connected)
    build_bottombar(frame, H, W, arduino_port)

    build_video_area(frame, cam_display, H, W,
                     faces_flipped, roi_flipped,
                     prediction_label, prediction_color)

    build_left_panel(frame, H, W,
                     eye_count, eye_previews,
                     prediction_label, prediction_color,
                     CONFIDENCE_THRESHOLD, conf_threshold_pct)

    build_right_panel(frame, H, W,
                      current_probs, current_probs,
                      confidence, eye_count, prediction_label)

    draw_prediction_chip(frame, H, W,
                         prediction_label, prediction_color, confidence)

    cv2.imshow(WIN_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(fname, frame)
        print(f"Screenshot disimpan: {fname}")

# Cleanup
print("\nMenutup aplikasi...")
if arduino and arduino.is_open:
    arduino.write(ARDUINO_CMD_OFF)  # Matikan LED saat keluar
    time.sleep(0.1)
    arduino.close()
    print("🔌 Arduino disconnected.")

cap.release()
cv2.destroyAllWindows()
print("✅  Aplikasi ditutup.")