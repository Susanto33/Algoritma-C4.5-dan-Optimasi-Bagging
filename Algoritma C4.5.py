import pandas as pd

# Implementasikan algoritma C4.5 (Anda dapat menggunakan library seperti scikit-learn)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Muat dataset
from google.colab import drive
drive.mount('/content/drive')

path_data = '/content/drive/My Drive/dataku.csv' # # Ubah sesuai dengan lokasi berkas Anda
df = pd.read_csv(path_data, sep=';')

# Bagi data menjadi fitur dan variabel target
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih pengklasifikasi pohon keputusan C4.5
clf = DecisionTreeClassifier(criterion='entropy') # Gunakan entropy untuk gain informasi
clf.fit(X_train, y_train)

# Lakukan prediksi pada set pengujian
y_pred = clf.predict(X_test)

# Evaluasi model
akurasi = accuracy_score(y_test, y_pred)
print("Akurasi:", akurasi)
