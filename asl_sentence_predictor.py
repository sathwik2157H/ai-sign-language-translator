import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
from datetime import datetime

# Load the trained model
model = load_model("asl_model_improved.h5")

# Categories used during training
CATEGORIES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

IMG_SIZE = 96

# Sentence construction helpers
predictions_queue = deque(maxlen=15)
current_sentence = ""

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_most_common_prediction(queue):
    if not queue:
        return None
    return max(set(queue), key=queue.count)

def update_sentence(pred):
    global current_sentence
    if pred == "space":
        current_sentence += " "
    elif pred == "del":
        current_sentence = current_sentence[:-1]
    elif pred == "nothing":
        pass
    else:
        current_sentence += pred

cap = cv2.VideoCapture(1)
last_pred = ""
last_time = time.time()
pred_cooldown = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    try:
        processed = preprocess(roi)
        prediction = model.predict(processed, verbose=0)
        class_index = np.argmax(prediction)
        pred_letter = CATEGORIES[class_index]
        predictions_queue.append(pred_letter)

        stable_pred = get_most_common_prediction(predictions_queue)

        if stable_pred != last_pred and time.time() - last_time > pred_cooldown:
            update_sentence(stable_pred)
            last_pred = stable_pred
            last_time = time.time()

    except Exception as e:
        pass

    # Draw ROI and show prediction info
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Current: {last_pred}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {current_sentence}", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('b'):
        current_sentence = current_sentence[:-1]
    elif key == ord('s'):
        with open("saved_sentences.txt", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {current_sentence}\n")
        current_sentence = ""

cap.release()
cv2.destroyAllWindows()
