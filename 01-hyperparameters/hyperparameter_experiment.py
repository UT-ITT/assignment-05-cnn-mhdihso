import os
import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RandomBrightness
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(SCRIPT_DIR, 'gesture_dataset_sample')
CONDITIONS = ['like', 'stop']
IMG_SIZE = 64
COLOR_CHANNELS = 3
BRIGHTNESS_VALUES = [0.0, 0.05, 0.1, 0.3, 0.5]


def preprocess_image(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


print("Loading dataset...")
annotations = {}
for condition in CONDITIONS:
    with open(os.path.join(PATH, '_annotations', f'{condition}.json')) as f:
        annotations[condition] = json.load(f)

images = []
labels = []
label_names = []

for condition in CONDITIONS:
    cond_dir = os.path.join(PATH, condition)
    for filename in os.listdir(cond_dir):
        if filename.startswith('.'):
            continue
        uid = filename.split('.')[0]
        img = cv2.imread(os.path.join(cond_dir, filename))
        if img is None:
            continue
        try:
            annotation = annotations[condition][uid]
        except KeyError:
            continue
        for i, bbox in enumerate(annotation['bboxes']):
            x1 = int(bbox[0] * img.shape[1])
            y1 = int(bbox[1] * img.shape[0])
            w = int(bbox[2] * img.shape[1])
            h = int(bbox[3] * img.shape[0])
            crop = img[max(0, y1):y1 + h, max(0, x1):x1 + w]
            if crop.size == 0:
                continue
            preprocessed = preprocess_image(crop)
            label = annotation['labels'][i]
            if label not in label_names:
                label_names.append(label)
            images.append(preprocessed)
            labels.append(label_names.index(label))

X = np.array(images, dtype='float32') / 255.0
y = np.array(labels)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
num_classes = len(label_names)

print(f"Dataset: {len(X_train)} train, {len(X_test)} test, classes: {label_names}")

results = []

for brightness in BRIGHTNESS_VALUES:
    print(f"\n--- Training with RandomBrightness = {brightness} ---")

    layers = [Input(shape=(IMG_SIZE, IMG_SIZE, COLOR_CHANNELS))]
    if brightness > 0:
        layers.append(RandomBrightness(factor=brightness))

    layers += [
        Conv2D(64, (9, 9), activation='leaky_relu', padding='same'),
        MaxPooling2D((4, 4), padding='same'),
        Conv2D(32, (5, 5), activation='leaky_relu', padding='same'),
        MaxPooling2D((3, 3), padding='same'),
        Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ]

    model = Sequential(layers)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    stop_early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=20, batch_size=8, verbose=1,
        callbacks=[stop_early]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    start = time.time()
    for _ in range(10):
        model.predict(X_test, verbose=0)
    inference_time = (time.time() - start) / 10

    results.append({
        'brightness': brightness,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'inference_time': inference_time,
        'history': history.history,
        'epochs_trained': len(history.history['loss'])
    })

    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Test Loss:      {test_loss:.4f}")
    print(f"  Inference Time: {inference_time:.4f}s")
    print(f"  Epochs Trained: {len(history.history['loss'])}")

print("\n\n========== RESULTS ==========")
print(f"{'Brightness':<12} {'Test Acc':<12} {'Test Loss':<12} {'Inf. Time':<12} {'Epochs':<8}")
for r in results:
    print(f"{r['brightness']:<12} {r['test_acc']:<12.4f} {r['test_loss']:<12.4f} {r['inference_time']:<12.4f}s {r['epochs_trained']:<8}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

vals = [r['brightness'] for r in results]
accs = [r['test_acc'] for r in results]
losses = [r['test_loss'] for r in results]
times = [r['inference_time'] for r in results]

axes[0].bar(range(len(vals)), accs, color=colors)
axes[0].set_xticks(range(len(vals)))
axes[0].set_xticklabels([str(v) for v in vals])
axes[0].set_xlabel('Random Brightness Factor')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Test Accuracy vs Random Brightness')
axes[0].set_ylim(0, 1.1)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

axes[1].bar(range(len(vals)), losses, color=colors)
axes[1].set_xticks(range(len(vals)))
axes[1].set_xticklabels([str(v) for v in vals])
axes[1].set_xlabel('Random Brightness Factor')
axes[1].set_ylabel('Test Loss')
axes[1].set_title('Test Loss vs Random Brightness')
for i, v in enumerate(losses):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

axes[2].bar(range(len(vals)), times, color=colors)
axes[2].set_xticks(range(len(vals)))
axes[2].set_xticklabels([str(v) for v in vals])
axes[2].set_xlabel('Random Brightness Factor')
axes[2].set_ylabel('Inference Time (s)')
axes[2].set_title('Inference Time vs Random Brightness')
for i, v in enumerate(times):
    axes[2].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'hyperparameter_results.png'), dpi=150)
plt.close()

rows = 2
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
for idx, r in enumerate(results):
    ax = axes[idx // cols][idx % cols]
    epochs = range(1, r['epochs_trained'] + 1)
    ax.plot(epochs, r['history']['accuracy'], 'b-', label='Train Acc')
    ax.plot(epochs, r['history']['val_accuracy'], 'r-', label='Val Acc')
    ax.plot(epochs, r['history']['loss'], 'b--', alpha=0.5, label='Train Loss')
    ax.plot(epochs, r['history']['val_loss'], 'r--', alpha=0.5, label='Val Loss')
    ax.set_title(f'Brightness = {r["brightness"]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy / Loss')
    ax.legend(loc='center right')
    ax.set_ylim(0, 1.5)

if len(results) < rows * cols:
    for idx in range(len(results), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

plt.suptitle('Training Curves for Different Random Brightness Intensities', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'training_curves.png'), dpi=150)
plt.close()

print("\nPlots saved:")
print(f"  - {os.path.join(SCRIPT_DIR, 'hyperparameter_results.png')}")
print(f"  - {os.path.join(SCRIPT_DIR, 'training_curves.png')}")

best = max(results, key=lambda r: r['test_acc'])
print(f"\n========== FINDINGS ==========")
print(f"Best Random Brightness factor: {best['brightness']} with test accuracy {best['test_acc']:.4f}")
print("\nDone!")
