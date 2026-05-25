"""
data_collection.py  (VERSI PERBAIKAN)
======================================
Perubahan utama:
- Deteksi DAN tampilkan kotak pada KEDUA mata
- Filter false positive: mata harus berada di SETENGAH ATAS ROI wajah
- Filter ukuran: mata terlalu kecil/besar diabaikan
- Simpan RATA-RATA dari kedua mata (lebih stabil) atau mata kiri saja
  tergantung pilihan Anda (lihat variabel USE_BOTH_EYES)
- Feedback visual lebih jelas
"""

import cv2
import os
import numpy as np

# ─── KONFIGURASI ───────────────────────────────────────────────────────────────
IMG_SIZE      = 64
DATASET_DIR   = "dataset"
CATEGORIES    = ["kiri", "kanan", "tengah"]
USE_BOTH_EYES = True   # True = simpan kedua mata sebagai 2 sampel terpisah
                       # False = hanya mata kiri (lebih konsisten untuk beberapa kasus)

for cat in CATEGORIES:
    os.makedirs(os.path.join(DATASET_DIR, cat), exist_ok=True)

# ─── LOAD HAAR CASCADE ─────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def preprocess_eye(eye_roi):
    """
    Pipeline preprocessing - HARUS IDENTIK di semua file.
    1. Resize ke 64x64
    2. CLAHE: normalisasi kontras lokal (atasi variasi cahaya)
    3. Adaptive Threshold: tonjolkan pupil sebagai area putih
    4. Normalisasi ke [0.0, 1.0]
    """
    resized   = cv2.resize(eye_roi, (IMG_SIZE, IMG_SIZE))
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(resized)
    thresh    = cv2.adaptiveThreshold(
        equalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    return thresh.astype(np.float32) / 255.0

def filter_eyes(eyes, roi_height, roi_width):
    """
    Filter deteksi mata palsu (alis, hidung, dll).

    Aturan filter:
    1. Mata hanya boleh ada di 60% ATAS ROI wajah
       (ROI sudah dipotong 60% atas wajah, tapi masih bisa ada noise)
    2. Ukuran mata: minimal 15% dan maksimal 50% lebar ROI
    3. Jika > 2 mata terdeteksi, ambil 2 yang paling besar

    Returns: list of (ex, ey, ew, eh) yang sudah difilter, max 2 item
    """
    valid = []
    min_size = int(roi_width * 0.12)   # Minimal 12% lebar wajah
    max_size = int(roi_width * 0.55)   # Maksimal 55% lebar wajah

    for (ex, ey, ew, eh) in eyes:
        # Filter 1: posisi Y harus di bagian atas ROI
        if ey > roi_height * 0.65:
            continue
        # Filter 2: ukuran masuk akal
        if ew < min_size or ew > max_size:
            continue
        valid.append((ex, ey, ew, eh))

    # Ambil maksimal 2 mata, urutkan berdasarkan area (terbesar dulu)
    valid = sorted(valid, key=lambda e: e[2] * e[3], reverse=True)
    return valid[:2]

def count_images(category):
    folder = os.path.join(DATASET_DIR, category)
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("=" * 55)
print("Eye Tracker - Data Collection  (VERSI PERBAIKAN)")
print("Tekan L=Kiri | R=Kanan | C=Tengah | Q=Keluar")
print("Tips: Usahakan kedua mata terlihat jelas di kamera")
print("=" * 55)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ─── DETEKSI WAJAH ────────────────────────────────────────────────────────
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    ready_eyes    = []   # List eye_frame yang siap disimpan
    detected_count = 0

    for (fx, fy, fw, fh) in faces[:1]:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (200, 120, 50), 2)

        # Ambil ROI 60% atas wajah (area mata)
        roi_h     = int(fh * 0.60)
        roi_gray  = gray[fy:fy+roi_h, fx:fx+fw]
        roi_color = frame[fy:fy+roi_h, fx:fx+fw]

        # ─── DETEKSI MATA ─────────────────────────────────────────────────────
        raw_eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,   # Lebih halus agar tidak melewatkan mata
            minNeighbors=6,     # Lebih ketat untuk kurangi false positive
            minSize=(25, 25)
        )

        # Filter dan ambil max 2 mata terbaik
        filtered_eyes = filter_eyes(raw_eyes, roi_h, fw) if len(raw_eyes) > 0 else []
        detected_count = len(filtered_eyes)

        for (ex, ey, ew, eh) in filtered_eyes:
            # Gambar kotak HIJAU pada setiap mata yang valid
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 80), 2)
            # Titik tengah mata
            cx, cy = ex + ew//2, ey + eh//2
            cv2.circle(roi_color, (cx, cy), 3, (0, 255, 80), -1)

            # Crop + preprocess area mata
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            processed  = preprocess_eye(eye_region)
            ready_eyes.append(processed)

        break  # Hanya proses wajah pertama

    # ─── PREVIEW DI POJOK KIRI ATAS ───────────────────────────────────────────
    # Tampilkan setiap mata yang terdeteksi sebagai preview kecil
    for i, eye_frame in enumerate(ready_eyes[:2]):
        preview = cv2.resize((eye_frame * 255).astype(np.uint8), (80, 80))
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        px = 10 + i * 90
        frame[10:90, px:px+80] = preview_bgr
        cv2.putText(frame, f"Mata {i+1}", (px, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # ─── STATUS DETEKSI ───────────────────────────────────────────────────────
    status_color = (0, 255, 80) if detected_count == 2 else (0, 165, 255) if detected_count == 1 else (0, 0, 220)
    status_text  = f"Mata terdeteksi: {detected_count}/2"
    cv2.putText(frame, status_text, (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # ─── COUNTER PER KATEGORI ─────────────────────────────────────────────────
    y_off = 150
    for cat in CATEGORIES:
        cv2.putText(frame, f"{cat}: {count_images(cat)}", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
        y_off += 17

    cv2.putText(frame, "L=Kiri | R=Kanan | C=Tengah | Q=Keluar",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    cv2.imshow("Data Collection - Eye Tracker", frame)

    # ─── KEYBOARD HANDLER ─────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if len(ready_eyes) > 0:
        save_category = None
        if key == ord('l'):
            save_category = "kiri"
        elif key == ord('r'):
            save_category = "kanan"
        elif key == ord('c'):
            save_category = "tengah"

        if save_category:
            if USE_BOTH_EYES:
                # Simpan SEMUA mata yang terdeteksi sebagai sampel terpisah
                # Ini menggandakan jumlah data dan membuat model lebih generalis
                for eye_frame in ready_eyes:
                    count     = count_images(save_category)
                    save_path = os.path.join(DATASET_DIR, save_category, f"{count:05d}.npy")
                    np.save(save_path, eye_frame)
                print(f"[SAVED] {save_category} — {len(ready_eyes)} mata ({count_images(save_category)} total)")
            else:
                # Hanya simpan mata pertama (lebih konsisten, lebih sedikit data)
                count     = count_images(save_category)
                save_path = os.path.join(DATASET_DIR, save_category, f"{count:05d}.npy")
                np.save(save_path, ready_eyes[0])
                print(f"[SAVED] {save_category}/{count:05d}.npy")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nKoleksi data selesai!")
for cat in CATEGORIES:
    print(f"  {cat}: {count_images(cat)} gambar")