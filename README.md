# Proyek Klasifikasi Gambar: CIFAR-10

Proyek machine learning untuk klasifikasi gambar menggunakan dataset CIFAR-10. Proyek ini mengimplementasikan neural network dengan TensorFlow/Keras untuk mengklasifikasikan 10 kategori objek.

## Informasi Proyek

- **Nama:** Alridho Tristan Satriawan
- **Email:** alridho.tristan@gmail.com
- **Tanggal:** 2026

## Deskripsi

Proyek ini menggunakan dataset CIFAR-10 yang berisi 60.000 gambar RGB berwarna berukuran 32x32 piksel dalam 10 kategori:
- Pesawat
- Mobil
- Burung
- Kucing
- Rusa
- Anjing
- Katak
- Kuda
- Kapal
- Truk

Model deep learning dibangun dan dilatih untuk mengklasifikasikan gambar-gambar ini dengan akurasi tinggi.

## Struktur Proyek

```
.
├── notebook.ipynb          # Notebook Jupyter dengan implementasi lengkap
├── requirements.txt        # Daftar dependencies Python
├── saved_model/            # Model TensorFlow yang sudah disimpan
│   ├── saved_model.pb
│   └── variables/
├── tfjs_model/             # Model dalam format TensorFlow.js
│   └── model.json
└── tflite/                 # Model dalam format TensorFlow Lite (mobile)
    ├── model.tflite
    └── label.txt
```

## Teknologi yang Digunakan

### Framework & Library Utama
- **TensorFlow/Keras** - Deep learning framework
- **scikit-learn** - Machine learning utilities
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **PIL/OpenCV** - Image processing
- **Matplotlib/Seaborn** - Visualization

### Model Architecture
- MobileNet atau DenseNet121 (transfer learning)
- Custom CNN (Convolutional Neural Network)
- Optimizers: Adam, RMSprop, SGD

## Instalasi

1. Clone atau download proyek ini

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pastikan Anda memiliki Python 3.7+ installed

## Penggunaan

### Menjalankan Notebook

1. Buka `notebook.ipynb` menggunakan Jupyter Notebook atau VS Code
2. Jalankan sel-sel secara berurutan untuk:
   - Import libraries dan konfigurasi
   - Load dan explore dataset CIFAR-10
   - Preprocessing dan augmentasi data
   - Membangun dan melatih model
   - Evaluasi performa model
   - Menyimpan model dalam berbagai format

### Menggunakan Model yang Sudah Dilatih

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('saved_model/')

# Lakukan prediksi
predictions = model.predict(image_data)
```

## Model Outputs

Proyek ini menghasilkan model dalam 3 format:

1. **SavedModel** (`saved_model/`) - Format native TensorFlow
2. **TensorFlow.js** (`tfjs_model/`) - Untuk web deployment
3. **TensorFlow Lite** (`tflite/`) - Untuk mobile deployment

## Fitur Implementasi

- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Augmentation (rotasi, noise, gamma adjustment)
- ✅ Transfer Learning
- ✅ Model Training & Validation
- ✅ Confusion Matrix & Classification Report
- ✅ Model Evaluation & Visualization
- ✅ Multi-format Model Export
- ✅ Callbacks (Early Stopping, Checkpoint, ReduceLROnPlateau)

## Requirements

Lihat `requirements.txt` untuk daftar lengkap dependencies. Library utama meliputi:
- tensorflow
- keras
- scikit-learn
- numpy
- pandas
- opencv-python
- pillow
- matplotlib
- seaborn

## Catatan

- Dataset CIFAR-10 akan otomatis diunduh saat pertama kali menjalankan notebook
- Model dilatih menggunakan GPU jika tersedia
- Hasil training dapat bervariasi tergantung seed random dan hardware

## Lisensi

Proyek ini dibuat sebagai tugas pembelajaran machine learning.

## Kontak

Untuk pertanyaan atau feedback, silakan hubungi melalui email: alridho.tristan@gmail.com
