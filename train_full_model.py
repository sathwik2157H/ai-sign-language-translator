import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# === CONFIG ===
DATADIR = "asl_alphabet_train/asl_alphabet_train"  # Adjust if different
CATEGORIES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing'
]
IMG_SIZE = 64
LIMIT = 500  # images per category

# === DATA PREP ===
print("Loading data...")
data = []
labels = []

for idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATADIR, category)
    image_files = os.listdir(folder_path)[:LIMIT]

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(idx)

print("Finished loading data.")

# === PREPROCESS ===
X = np.array(data) / 255.0
y = to_categorical(labels, num_classes=len(CATEGORIES))

# === SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL ===
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# === TRAIN ===
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=2
)

# === SAVE MODEL ===
model.save("asl_model_full.h5")
print("✅ Model saved as asl_model_full.h5")
