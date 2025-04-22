import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Modeli yükle
model = tf.keras.models.load_model('8model.h5')

#
img_path = '5.jpg'  # Tahmin edilecek görsel
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle
img_array /= 255.0  # Normalizasyon

# Tahmin yap
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])

class_names = ['araba','gemi','kasa','kedi', 'kopek',"kulaklik","telofon","ucak"]  

predicted_class = class_names[class_idx]
confidence = predictions[0][class_idx]

print(f"Tahmin: {predicted_class} (%{confidence*100:.2f} doğruluk)")