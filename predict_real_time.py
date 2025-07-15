import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("asl_model.h5")

# Categories used during training
CATEGORIES = ['A', 'B', 'C', 'D', 'E']

IMG_SIZE = 64

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip to act like mirror
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI) for hand
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI
    try:
        processed = preprocess(roi)
        prediction = model.predict(processed, verbose=0)
        class_index = np.argmax(prediction)
        letter = CATEGORIES[class_index]
    except Exception as e:
        letter = "..."

    # Draw ROI and prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediction: {letter}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
