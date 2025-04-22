# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# # GPU kontrolü
# print("TensorFlow version:", tf.__version__)
# print("GPU Available:", tf.test.is_gpu_available())

# # Dosya yolları
# train_dir = "data/train"
# test_dir = "data/test"

# # Görsel boyutu ve batch
# img_height, img_width = 150, 150
# batch_size = 4

# # Verileri hazırla
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_gen = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical'
# )

# test_gen = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical'
# )

# # CNN Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     MaxPooling2D(2, 2),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(train_gen.num_classes, activation='softmax')
# ])

# # Modeli derle
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.0001),  # 'lr' yerine 'learning_rate' kullanılıyor
#     metrics=['accuracy']
# )

# # Eğitimi başlat
# history = model.fit(
#     train_gen,
#     steps_per_epoch=train_gen.samples // batch_size,
#     epochs=10,
#     validation_data=test_gen,
#     validation_steps=test_gen.samples // batch_size
# )

# # Eğitilen modeli kaydet
# model.save("kedi_kopek_modeli.h5")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# GPU kontrolü
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())

# Dosya yolları
train_dir = "data/train"
test_dir = "data/test"

# Görsel boyutu ve batch
img_height, img_width = 150, 150
batch_size = 16

# Verileri hazırla
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')  # Sınıf sayısı burada değiştirilebilir
])

# Modeli derle
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Eğitimi başlat
model.fit(
    train_gen,
    epochs=20,
    validation_data=test_gen
)

# Eğitilen modeli kaydet
model.save("8model.h5")
