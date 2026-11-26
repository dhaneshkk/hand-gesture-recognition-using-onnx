[[Japanese](./README.md) / English]

---
# hand-gesture-recognition-using-onnx
[Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) This repository replaces the entire MediaPipe implementation with ONNX, borrowing from [@Kazuhito00](https://github.com/Kazuhito00). This is a sample program that recognizes hand signs and finger gestures using a simple MLP. The only thing I've confirmed so far is that it works.

https://user-images.githubusercontent.com/33194443/189632510-1823cd54-bc36-4889-ac14-adc16deba9b8.mp4

This repository contains the following content:
- Sample Program
- Hand Detection Model (Modified ONNX)
- Palm Landmark Detection Model (Modified ONNX)
- Hand Sign Recognition Model (Modified ONNX)
- Finger Gesture Recognition Model (Modified ONNX)
- Training Data and Training Notebook for Hand Sign Recognition
- Training Data and Training Notebook for Finger Gesture Recognition

# Requirements
- onnxruntime 1.12.0 or onnxruntime-gpu 1.12.0
- opencv-contrib-python 4.6.0.66 or Later
- Tensorflow 2.10.0 (Only if you want to recreate ONNX files after training)
- PyTorch 1.12.0 (Only if you want to recreate ONNX files after training)
- tf2onnx 1.12.0 or Later (Only if you want to recreate ONNX files after training)
- simple-onnx-processing-tools 1.0.54 or Later (Only if you want to recreate the ONNX file after training)
- scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix during training)
- matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix during training)

# Demo
To run the demo using a webcam:
```bash
python app.py
```

The following options can be specified when running the demo.
```
--device
Specify the camera device number (default: 0)

--width
Width of camera capture (default: 640)

--height
Height of camera capture (default: 480)

--min_detection_confidence
Detection confidence threshold (default: 0.6)

--disable_image_flip
Disables horizontal flipping of input image
```

# Directory
```
.
│ app.py
│ keypoint_classification.ipynb
│ point_history_classification.ipynb
│ requirements.txt
│ README.md
│
├─model
│ ├─keypoint_classifier
│ │ │ tflite_to_onnx.sh
│ │ │ make_argmax.py
│ │ │ keypoint.csv
│ │ │ keypoint_classifier.hdf5
│ │ │ keypoint_classifier.py
│ │ │ keypoint_classifier.tflite
│ │ │ keypoint_classifier.onnx
│ │ └─ keypoint_classifier_label.csv
│ │
│ └─ point_history_classifier
│ │ tflite_to_onnx.sh
│ │ make_argmax.py
│ │ point_history.csv
│ │ point_history_classifier.hdf5
│ │ point_history_classifier.py
│ │ point_history_classifier.tflite
│ │ point_history_classifier.onnx
│ └─ point_history_classifier_label.csv
│
└─utils
│ cvfpscalc.py
└─ utils.py
```
### app.py
This is a sample program for inference. <br>You can also collect training data (keypoints) for hand sign recognition,<br>
and training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
Model training script for hand sign recognition.

### point_history_classification.ipynb
Model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition. <br>
The following files are stored here.
- tflite ONNX conversion script (tflite_to_onnx.sh)
- ONNX component generation program (make_argmax.py)
- Training data (keypoint.csv)
- Trained model (keypoint_classifier.tflite)
- Trained model (keypoint_classifier.onnx)
- Label data (keypoint_classifier_label.csv)
- Inference classes (keypoint_classifier.py)

### model/point_history_classifier
This directory contains files related to finger gesture recognition. <br>
The following files are stored here.
- tflite ONNX conversion script (tflite_to_onnx.sh)
- ONNX component generation program (make_argmax.py)
- Training data (point_history.csv)
- Trained model (point_history_classifier.tflite)
- Trained model (point_history_classifier.onnx)
- Label data (point_history_classifier_label.csv)
- Inference class (point_history_classifier.py)

### utils/cvfpscalc.py
Module for FPS measurement.

### utils/utils.py
Functions for image processing.

# Training
For hand sign recognition and finger gesture recognition, you can add or change training data and retrain the model.

### Hand Sign Recognition Training Method
#### 1. Training Data Collection
Pressing "k" will enter keypoint saving mode ("MODE: Logging Key Point" will be displayed).<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
Pressing "0" through "9" will add keypoints to "model/keypoint_classifier/keypoint.csv" as shown below. <br>
Column 1: Pressed number (used as class ID), Column 2: TrackID, Columns 3 and beyond: Keypoint coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
Keypoint coordinates are saved after the following preprocessing steps have been performed up to ④. <br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
By default, three types of training data are included: paper (class ID: 0), rock (class ID: 1), and pointing hand (class ID: 2). <br>
Add classes from 3 onward as needed, or delete existing data from the CSV file to prepare your own training data. <br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2. Model Training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and run it in order from top to bottom. <br>
To change the number of classes in the training data, change the value of "NUM_CLASSES = 3" and modify the labels in "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate. <br><br>

#### X. Model Structure
The following is an image of the model provided in "[keypoint_classification.ipynb](keypoint_classification.ipynb)".
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"

# Fork changes (dhaneshkk)

### Fixes in `app.py` and the model wrappers
- **One-to-one hand tracking** (`utils/tracker.py`): global greedy matching, so two hands can no longer share a track ID. Tracks that go `--track_max_missed` frames without a match are dropped, together with their fingertip and gesture history.
- **Per-hand data stays paired**: `HandLandmark` returns `keep_indices`, and landmarks, signs, fingertip histories, gesture votes and labels are all keyed by track ID. Before, a low-scoring hand or a change in detection order could attach one hand's data to another's box. The fingertip history is also updated before classification, removing a one-frame lag.
- **Training-data logging matches the notebooks**: rows are `label, 42 features` (keypoint) and `label, 32 features` (history). The old logger added a `trackid` column, which the notebooks then read as a feature. Every hand is logged, and point histories are only logged once they are full length.
- **Geometry uses the real frame size**, not `--width/--height`, and portrait frames are handled (padding correction on both axes). Rotation angles stay as floats.
- **The video writer** is created from the first real frame. Use `-o/--output` to set the file, `-nv/--no_video` to skip recording, and `--headless` to run without a window.
- Model and CSV paths are resolved relative to the repo, so the app can be started from any directory. Fixed NumPy 2.x crashes in `HandLandmark`, and CUDA warnings no longer appear on CPU-only installs.

### Retraining with SAM + MCC
[`train_sam_mcc.ipynb`](train_sam_mcc.ipynb) is a single Colab cell. It retrains both classifiers with **Sharpness-Aware Minimization** (rho=0.05, wrapping Adam) on a stratified 70/15/15 split, picks the checkpoint by **validation MCC**, reports **test MCC** (checked against scikit-learn) with confusion matrices, and exports ONNX models with the same interface as the shipped ones.

| model | test MCC, SAM retrain | test MCC, shipped model* |
|---|---|---|
| keypoint (3 classes, n=668) | 0.955 | 0.951 |
| point history LSTM (4 classes, n=795) | 0.953 | 0.975 |

\* The shipped models were trained on the same CSVs with a different split, so part of this test set was probably in their training data, and their numbers are optimistic.

```bash
python app.py \
  --keypoint_model model/keypoint_classifier/keypoint_classifier_sam.onnx \
  --point_history_model model/point_history_classifier/point_history_classifier_lstm_sam.onnx
```

### Verification
- `python -m pytest -q tests`: tracker invariants (randomised), CSV schema against the notebook loaders, and end-to-end landmarks on `hand.png` checked across resolutions and portrait rotation.
- `proofs/TrackerInjective.lean` (Lean 4, core only) proves the tracker's greedy matching is one-to-one and only returns candidate pairs. CI checks it (`.github/workflows/ci.yml`).

