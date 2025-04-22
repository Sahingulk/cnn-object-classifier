
# 🧠  Evrişimli Sinir Ağı (CNN) ile 8 Sınıflı Görüntü Sınıflandırma

## 📌 Proje Açıklaması

Bu projede, temel CNN (Convolutional Neural Network) mimarisi kullanılarak 8 farklı sınıfa ait nesnelerin görüntüleri sınıflandırılmıştır. Model, TensorFlow ve Keras kütüphaneleri ile sıfırdan manuel olarak inşa edilmiştir. Eğitim süreci boyunca CUDA destekli GPU kullanılarak hızlandırma sağlanmıştır.

## 🎯 Amaç

- Derin öğrenme algoritmalarını bireysel bilgisayarda uygulamalı olarak öğrenmek.
- YOLO gibi hazır modelleri kullanmadan temel CNN mimarisi oluşturmak.
- Gerçek görüntülerle model eğitmek, test etmek ve başarı oranlarını yorumlamak.
- Birden fazla nesne içeren görüntülerde başarıyı gözlemlemek.

## 📁 Veri Seti

- Toplam **8 farklı nesne sınıfı** kullanılmıştır.
- Her sınıf için `train/` ve `test/` klasörlerinde ayrı ayrı görseller bulunmaktadır.
- Görseller `150x150 piksel` boyutuna ölçeklendirilmiştir.
- Örnek sınıflar: `kedi`, `köpek`, `araba`, `bisiklet`, `uçak`, `gemi`, `kuş`, `saat`

## ⚙️ Kullanılan Teknolojiler

| Teknoloji         | Sürüm       |
|------------------|-------------|
| Python           | 3.7         |
| TensorFlow       | 1.15        |
| CUDA             | 10.0        |
| cuDNN            | 7.6.5       |
| Keras            | Dahili      |
| GPU              | NVIDIA RTX (kullanıcı bilgisayarına bağlı) |

## 🧱 Model Mimarisi

```
Input Layer: 150x150x3

[Conv2D]        -> 32 filtre, 3x3, ReLU  
[MaxPooling2D]  -> 2x2

[Conv2D]        -> 64 filtre, 3x3, ReLU  
[MaxPooling2D]  -> 2x2

[Flatten]       
[Dense]         -> 64 nöron, ReLU  
[Dropout]       -> %50  
[Output]        -> 8 sınıf için Softmax
```

## 📊 Eğitim Sonuçları

- Eğitim Epoch sayısı: 10  
- Batch size: 16  
- Eğitim ve test doğrulukları grafiklerle aşağıda gösterilmiştir:

<p align="center">
  <img src="grafikler/train_vs_val_accuracy.png" width="400"/>
  <br>
  <b>Şekil 1.</b> Eğitim ve doğrulama doğruluğu
</p>

<p align="center">
  <img src="grafikler/train_vs_val_loss.png" width="400"/>
  <br>
  <b>Şekil 2.</b> Eğitim ve doğrulama kaybı
</p>

## 🚀 Canlı Kullanım (Inference)

Model başarıyla eğitildikten sonra `.h5` formatında kaydedilmiştir. Aşağıdaki gibi yüklenerek yeni görüntüler üzerinde sınıflandırma yapılabilir:

```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model("model/cnn_model.h5")

img = image.load_img("test_img.jpg", target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("Tahmin edilen sınıf:", predicted_class)
```

## 👨‍💻 Geliştirici

| Ad Soyad     | Üniversite | Bölüm                      |
|--------------|------------|----------------------------|
| Şahin Gül    | BTÜ        | Bilgisayar Mühendisliği    |
