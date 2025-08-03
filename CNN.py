import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ขนาดภาพ
IMG_SIZE = 64

# Path dataset
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# สร้าง ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลด dataset
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary'
)

# สร้าง CNN โมเดล
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output: 0 = closed, 1 = open
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# เทรน!
model.fit(train_gen, validation_data=test_gen, epochs=10)

# บันทึกโมเดล
model.save("my_model.keras")
print("โมเดลถูกบันทึกไว้เป็น eye_state_cnn.h5")
