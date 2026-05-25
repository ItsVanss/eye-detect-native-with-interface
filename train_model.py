"""
train_model.py
==============
Tujuan: Membangun dan melatih CNN untuk mengklasifikasikan
arah pandangan mata: Kiri, Kanan, atau Tengah.

Konsep CNN yang Digunakan:
- Conv2D: layer konvolusi mendeteksi fitur lokal (tepi, sudut, tekstur)
  Setiap filter adalah matriks kecil yang "digeser" di atas gambar.
  Filter belajar mendeteksi pola spesifik selama training.
- MaxPooling2D: mereduksi dimensi spasial, membuat fitur lebih invariant
  terhadap translasi kecil (posisi pupil bergeser sedikit = tetap terdeteksi)
- Dropout: secara acak "mematikan" neuron selama training untuk
  mencegah overfitting (hafalan berlebihan)
- Dense + Softmax: layer klasifikasi akhir, output berupa probabilitas
  untuk setiap kelas (jumlahnya selalu = 1.0)
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ─── KONFIGURASI ───────────────────────────────────────────────────────────────
IMG_SIZE   = 64
DATASET_DIR = "dataset"
MODEL_PATH  = "model/eye_tracker.h5"
CATEGORIES  = ["kiri", "kanan", "tengah"]  # Urutan = label index 0, 1, 2
EPOCHS      = 30
BATCH_SIZE  = 32
os.makedirs("model", exist_ok=True)

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Memuat dataset...")
X, y = [], []

for label_idx, category in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_DIR, category)
    files  = [f for f in os.listdir(folder) if f.endswith(".npy")]
    print(f"  {category}: {len(files)} sampel (label={label_idx})")

    for fname in files:
        img = np.load(os.path.join(folder, fname))  # Shape: (64, 64)
        X.append(img)
        y.append(label_idx)

X = np.array(X)  # Shape: (N, 64, 64)
y = np.array(y)  # Shape: (N,)

# Keras Conv2D mengharapkan format (batch, height, width, channels)
# Tambahkan dimensi channel: (N, 64, 64) → (N, 64, 64, 1)
X = X[..., np.newaxis]

print(f"\nTotal sampel: {len(X)}")
print(f"Shape input : {X.shape}")
print(f"Distribusi  : {np.bincount(y)} (kiri, kanan, tengah)")

# ─── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
# test_size=0.2   → 20% data untuk evaluasi (tidak digunakan saat training)
# stratify=y      → pastikan proporsi kelas sama di train dan test set
# random_state=42 → agar hasil reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ─── DATA AUGMENTATION ─────────────────────────────────────────────────────────
# Augmentasi meningkatkan variasi data training tanpa mengumpulkan data baru.
# Ini membantu model lebih robust terhadap variasi posisi kepala dan pencahayaan.
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,       # Putar ±10 derajat (simulasi miring kepala)
    width_shift_range=0.1,   # Geser horizontal ±10% (posisi mata berubah)
    height_shift_range=0.1,  # Geser vertikal ±10%
    zoom_range=0.1,           # Zoom in/out ±10% (jarak ke kamera berubah)
    horizontal_flip=False,    # JANGAN flip! Kiri ≠ Kanan untuk klasifikasi ini
    brightness_range=[0.7, 1.3]  # Variasi kecerahan (simulasi pencahayaan berbeda)
)
datagen.fit(X_train)

# ─── ARSITEKTUR CNN ────────────────────────────────────────────────────────────
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=3):
    """
    Sequential CNN: data mengalir linear dari satu layer ke layer berikutnya.
    
    Blok 1 (Conv 32):
      - 32 filter berukuran 3x3 belajar fitur dasar: tepi pupil, iris
      - BatchNormalization: normalisasi output tiap layer agar training stabil
      - MaxPooling: kurangi 64x64 → 31x31

    Blok 2 (Conv 64):
      - 64 filter menggabungkan fitur dasar menjadi pola lebih kompleks
      - MaxPooling: 31x31 → 14x14
      
    Blok 3 (Conv 128):
      - 128 filter mendeteksi pola high-level: arah pandang kiri/kanan
      - MaxPooling: 14x14 → 6x6
      
    Flatten → Dense → Output:
      - Flatten: ubah tensor 3D → vektor 1D (6x6x128 = 4608 nilai)
      - Dense(128): lapisan fully-connected untuk kombinasi fitur global
      - Dropout(0.5): 50% neuron dimatikan saat training → cegah overfitting
      - Dense(3, softmax): output probabilitas untuk 3 kelas
    """
    model = models.Sequential([
        # Blok Konvolusi 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok Konvolusi 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok Konvolusi 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Klasifikasi
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),          # Dropout hanya aktif saat training
        layers.Dense(num_classes, activation='softmax')  # Output = probabilitas
    ])
    return model

model = build_model()
model.summary()

# ─── COMPILE MODEL ─────────────────────────────────────────────────────────────
# Adam: optimizer adaptif, lr=0.001 adalah default yang umumnya baik
# sparse_categorical_crossentropy: cocok untuk label integer (0, 1, 2)
#   vs categorical_crossentropy yang butuh one-hot encoding
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─── CALLBACKS ─────────────────────────────────────────────────────────────────
callbacks = [
    # Hentikan training jika val_loss tidak membaik selama 5 epoch
    # Cegah overfitting dan hemat waktu training
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),

    # Simpan model terbaik berdasarkan val_accuracy
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

# ─── TRAINING ──────────────────────────────────────────────────────────────────
print("\nMemulai training...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUASI ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy : {acc*100:.2f}%")
print(f"   Test Loss     : {loss:.4f}")
print(f"   Model disimpan: {MODEL_PATH}")

# ─── PLOT TRAINING CURVE ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='Train Acc')
ax1.plot(history.history['val_accuracy'], label='Val Acc')
ax1.set_title('Akurasi Training')
ax1.legend()
ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss Training')
ax2.legend()
plt.tight_layout()
plt.savefig("model/training_curve.png")
plt.show()
print("Kurva training disimpan ke model/training_curve.png")