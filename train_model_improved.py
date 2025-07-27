import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 96  # MobileNetV2 expects at least 96x96
DATA_DIR = "asl_alphabet_train/asl_alphabet_train"
CATEGORIES = sorted(os.listdir(DATA_DIR))  # Must include all 29 signs (A-Z + space/del/nothing)

print("Loading and preprocessing data...")
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    for img_name in os.listdir(path)[:500]:  # limit per class to 500 (you can increase)
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(CATEGORIES.index(category))
        except Exception as e:
            print(f"Error loading image: {e}")

X = np.array(data) / 255.0
y = to_categorical(labels, num_classes=len(CATEGORIES))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
# MobileNetV2 Base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base model

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(CATEGORIES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
    verbose=1
)
model.save("asl_model_improved.h5")
print("✅ Improved model saved as asl_model_improved.h5")
from sklearn.metrics import classification_report
import numpy as np

# Predict on validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)


