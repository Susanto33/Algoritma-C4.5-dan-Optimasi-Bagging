# Install library yang diperlukan
!pip install numpy pandas scikit-learn

# Impor library yang dibutuhkan
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Muat dataset
file_path = '/content/drive/MyDrive/dataku.csv'  # Ubah sesuai dengan lokasi berkas Anda
df = pd.read_csv(file_path, sep=';')

# Pra-pemrosesan (jika diperlukan)
# Diasumsikan kolom terakhir adalah variabel target
X = df.iloc[:, :-1]  # Fitur
y = df.iloc[:, -1]   # Variabel target

# Bagi dataset menjadi set data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementasi algoritma C4.5 dengan Bagging
base_classifier = DecisionTreeClassifier(criterion='entropy') # Menggunakan entropi untuk gain informasi
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Latih model
bagging_classifier.fit(X_train, y_train)

# Prediksi
y_pred = bagging_classifier.predict(X_test)

# Hitung akurasi
akurasi = accuracy_score(y_test, y_pred)
print("Akurasi:", akurasi)

# Hitung error (tingkat kesalahan)
error = 1 - akurasi
print("Error:", error)

# Hitung presisi untuk kelas positif (YES = 2)
presisi_positif = precision_score(y_test, y_pred, pos_label=2)
print("Presisi untuk kelas positif:", presisi_positif)

# Hitung recall untuk kelas positif (YES = 2)
recall_positif = recall_score(y_test, y_pred, pos_label=2)
print("Recall untuk kelas positif:", recall_positif)
