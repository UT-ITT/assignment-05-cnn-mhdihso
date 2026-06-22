import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PATH = os.path.join(REPO_ROOT, '01-hyperparameters', 'gesture_dataset_sample')

CLASSES = ['like', 'rock', 'peace', 'no_gesture']
IMG_SIZE = 64


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


# =============================================
# STEP 1: Capture images and annotate with camera
# =============================================
print("=== STEP 1: Capture your hand gesture images ===")
annotations = {}

for gesture in ['like', 'rock', 'peace']:
    print(f"\nShow your '{gesture}' gesture. Press SPACE to take photo, Q to quit.")
    cap = cv2.VideoCapture(0)

    photo = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Gesture: {gesture} - SPACE to capture",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            photo = frame.copy()
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    if photo is None:
        print("No photo taken, skipping.")
        continue

    img_name = f"mehdi_{gesture}.jpg"
    img_path = os.path.join(SCRIPT_DIR, img_name)
    cv2.imwrite(img_path, photo)
    print(f"Saved {img_name}")

    print(f"Draw a box around your '{gesture}' hand. Press ENTER to confirm.")
    roi = cv2.selectROI("Select hand", photo, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    h, w = photo.shape[:2]
    x, y, rw, rh = roi
    bbox = [x / w, y / h, rw / w, rh / h]

    annotations[f"mehdi_{gesture}"] = {
        "bboxes": [bbox],
        "labels": [gesture],
        "landmarks": [],
        "leading_conf": 1.0,
        "leading_hand": "right",
        "user_id": ""
    }

annot_path = os.path.join(SCRIPT_DIR, "annot-mehdi.json")
with open(annot_path, "w") as f:
    json.dump(annotations, f, indent=4)
print(f"\nAnnotations saved to annot-mehdi.json")


# =============================================
# STEP 2: Train CNN on HaGRID subset
# =============================================
print("\n=== STEP 2: Training CNN ===")
X_train = []
y_train = []

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
                    X_train.append(crop)
                    y_train.append(CLASSES.index(label))

X_train = np.array(X_train, dtype='float32') / 255.0
y_train = np.array(y_train)
print(f"Loaded {len(X_train)} training samples")

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
model.fit(X_train, y_train, epochs=10, batch_size=32)


# =============================================
# STEP 3: Evaluate on user + tutor images
# =============================================
print("\n=== STEP 3: Evaluating ===")
X_eval = []
y_true = []

tutor_dir = os.path.join(SCRIPT_DIR, 'tutor_images')
with open(os.path.join(tutor_dir, 'annot-tutors.json')) as f:
    tutor_annots = json.load(f)
for uid, annot in tutor_annots.items():
    img_path = os.path.join(tutor_dir, f'{uid}.png')
    if not os.path.exists(img_path):
        img_path = os.path.join(tutor_dir, f'{uid}.jpg')
    for i, bbox in enumerate(annot['bboxes']):
        label = annot['labels'][i]
        if label in CLASSES:
            crop = load_and_crop(img_path, bbox)
            if crop is not None:
                X_eval.append(crop)
                y_true.append(CLASSES.index(label))

with open(annot_path) as f:
    user_annots = json.load(f)
for uid, annot in user_annots.items():
    img_path = os.path.join(SCRIPT_DIR, f'{uid}.jpg')
    if not os.path.exists(img_path):
        img_path = os.path.join(SCRIPT_DIR, f'{uid}.png')
    for i, bbox in enumerate(annot['bboxes']):
        label = annot['labels'][i]
        if label in CLASSES:
            crop = load_and_crop(img_path, bbox)
            if crop is not None:
                X_eval.append(crop)
                y_true.append(CLASSES.index(label))

X_eval = np.array(X_eval, dtype='float32') / 255.0
y_true = np.array(y_true)

y_pred = np.argmax(model.predict(X_eval), axis=1)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()

out_path = os.path.join(SCRIPT_DIR, 'conf-matrix.png')
plt.savefig(out_path)
plt.close()
print(f"\nDone! Confusion matrix saved to {out_path}")
