# Assignment 5 - Convolutional Neural Networks

## Setup

```bash
pip install -r requirements.txt
```

---

## 1. Exploring Hyperparameters (5P)

**Hyperparameter: Random Brightness Intensity**

I explored how the `RandomBrightness` data augmentation layer affects CNN performance. This layer randomly shifts pixel brightness during training to simulate different lighting conditions.

**What I did:**
- Used the CNN from the course notebook and the HaGRID subset dataset (like, stop gestures)
- Trained 5 separate models, each with a different brightness factor: 0.0, 0.05, 0.1, 0.3, 0.5
- Kept all other hyperparameters constant (same architecture, optimizer, batch size, etc.)
- Measured test accuracy, test loss, and inference time for each model
- Plotted bar charts comparing the results and training curves for each value

**How to run:**
```bash
python 01-hyperparameters/hyperparameter_experiment.py
```

**Results:**
- Brightness = 0.0 (no augmentation) performed best with 90.62% accuracy
- Any brightness augmentation significantly reduced accuracy (down to ~31-42%)
- The dataset is too small and sensitive to brightness changes for this augmentation to help
- Inference time is not affected by brightness augmentation

**Files:**
- `01-hyperparameters/hyperparameters.ipynb` — notebook with approach, assumptions, and findings
- `01-hyperparameters/hyperparameter_experiment.py` — experiment script
- `01-hyperparameters/hyperparameter_results.png` — bar chart results
- `01-hyperparameters/training_curves.png` — training/validation curves

---

## 2. Gathering a Dataset (4P)

I extended the HaGRID dataset by capturing my own hand gesture images for three categories: like, rock, and peace.

**What I did:**
- Captured one image per gesture (like, rock, peace) using my webcam
- Used OpenCV's `selectROI` to draw bounding boxes around each hand in the images
- Saved the annotations in `annot-mehdi.json` in the same format as the original HaGRID dataset (bboxes as normalized [0-1] coordinates, labels, landmarks, etc.)
- Trained a simple CNN on the HaGRID subset (like, rock, peace, no_gesture)
- Ran predictions on both my images and the tutors' images
- Plotted the combined confusion matrix and saved it as `conf-matrix.png`

**How to run:**
```bash
python 02-dataset/assignment2.py
```

**Files:**
- `02-dataset/assignment2.py` — script for capturing, training, and evaluation
- `02-dataset/annot-mehdi.json` — my hand gesture annotations (HaGRID-compatible)
- `02-dataset/mehdi_like.jpg`, `mehdi_rock.jpg`, `mehdi_peace.jpg` — my captured images
- `02-dataset/conf-matrix.png` — confusion matrix (user + tutor images)
- `02-dataset/tutor_images/` — tutor images and their annotations

---

## 3. Gesture-based Media Controls (6P)

I built a real-time gesture-based media controller that uses the webcam to recognize hand gestures and trigger media playback controls.

**What I did:**
- Trained a CNN classifier on the HaGRID subset to distinguish like, rock, peace, and no_gesture
- Built a live webcam application that continuously captures frames
- Used skin color detection (HSV filtering) to check if a hand is present in the detection area
- The CNN classifies the detected hand gesture in real-time
- Mapped gestures to media controls using the `pynput` library:
  - **Like (thumbs up)** → Play/Pause
  - **Rock** → Skip to next track
  - **Peace (V sign)** → Volume up
  - **No hand / no gesture** → Do nothing (handles the "no gesture" case)
- Added a 1.5 second cooldown between actions to prevent repeated triggering
- The model is trained once and saved, so subsequent runs load it instantly for low latency

**How to run:**
```bash
python 03-media_control/media_control.py
```

**Tips:**
- Hold your hand close to the camera, filling the detection box
- Use a plain background (white wall) for best results
- Press Q to quit
- On macOS, allow camera and accessibility permissions in System Preferences

**Files:**
- `03-media_control/media_control.py` — gesture media controller script
