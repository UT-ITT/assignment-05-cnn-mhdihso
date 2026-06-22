import os
import cv2
import json
import time
import numpy as np
from pynput.keyboard import Key, Controller
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PATH = os.path.join(REPO_ROOT, '01-hyperparameters', 'gesture_dataset_sample')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gesture_model.keras')

CLASSES = ['like', 'rock', 'peace', 'no_gesture']
IMG_SIZE = 64
keyboard = Controller()

# Gesture to media control mapping
GESTURE_ACTIONS = {
    'like':  'Play/Pause',
    'rock':  'Next Track',
    'peace': 'Volume Up',
}

COOLDOWN = 1.5  # seconds between actions


def load_and_crop(img_path, bbox):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0] * w))
    y1 = max(0, int(bbox[1] * h))
    x2 = min(w, int((bbox[0] + bbox[2]) * w))
    y2 = min(h, int((bbox[1] + bbox[3]) * h))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


def train_model():
    """Train CNN on the HaGRID subset."""
    print("Training model...")
    X, y = [], []

    for condition in ['like', 'rock', 'peace']:
        annot_file = os.path.join(PATH, '_annotations', f'{condition}.json')
        with open(annot_file) as f:
            annots = json.load(f)
        cond_dir = os.path.join(PATH, condition)
        for filename in os.listdir(cond_dir):
            if filename.startswith('.'):
                continue
            uid = filename.split('.')[0]
            if uid not in annots:
                continue
            img_path = os.path.join(cond_dir, filename)
            for i, bbox in enumerate(annots[uid]['bboxes']):
                label = annots[uid]['labels'][i]
                if label in CLASSES:
                    crop = load_and_crop(img_path, bbox)
                    if crop is not None:
                        X.append(crop)
                        y.append(CLASSES.index(label))

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    print(f"Loaded {len(X)} training samples")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def detect_hand_by_skin(frame, roi):
    """Simple skin color detection to check if a hand is present in the ROI."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Skin color range in HSV
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(mask > 0) / mask.size
    return skin_ratio > 0.15  # at least 15% skin pixels


def do_media_action(gesture):
    """Trigger the media key for the given gesture."""
    if gesture == 'like':
        keyboard.press(Key.media_play_pause)
        keyboard.release(Key.media_play_pause)
    elif gesture == 'rock':
        keyboard.press(Key.media_next)
        keyboard.release(Key.media_next)
    elif gesture == 'peace':
        for _ in range(5):
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)


def main():
    # Load or train model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    else:
        model = train_model()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    last_action_time = 0
    current_gesture = 'no_gesture'

    print("\n=== Gesture Media Controller ===")
    print("Hold your hand CLOSE to the camera, filling the green box.")
    print("  like  (thumbs up) -> Play/Pause")
    print("  rock              -> Next Track")
    print("  peace (V sign)    -> Volume Up")
    print("  no hand           -> Nothing")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ROI: right side of screen where you hold your hand
        box_w = 250
        box_h = 250
        x1 = w - box_w - 20
        y1 = h // 2 - box_h // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        roi = frame[y1:y2, x1:x2]
        gesture = 'no_gesture'
        confidence = 0.0

        # Check if there is a hand (skin) in the ROI
        hand_present = detect_hand_by_skin(frame, roi)

        if hand_present:
            resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            input_data = np.expand_dims(resized.astype('float32') / 255.0, axis=0)
            pred = model.predict(input_data, verbose=0)
            class_idx = np.argmax(pred)
            confidence = pred[0][class_idx]
            gesture = CLASSES[class_idx]

        # Trigger action with cooldown
        now = time.time()
        if gesture != 'no_gesture' and confidence > 0.70 and (now - last_action_time) > COOLDOWN:
            do_media_action(gesture)
            current_gesture = gesture
            last_action_time = now
            print(f"  -> {GESTURE_ACTIONS[gesture]}")
        elif gesture == 'no_gesture':
            current_gesture = 'no_gesture'

        # Draw UI
        color = (0, 255, 0) if current_gesture == 'no_gesture' else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if hand_present:
            label = f"{gesture} ({confidence:.0%})"
        else:
            label = "No hand detected"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        action = GESTURE_ACTIONS.get(current_gesture, "Waiting...")
        cv2.putText(frame, action, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Gesture Media Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Controller stopped.")


if __name__ == "__main__":
    main()
