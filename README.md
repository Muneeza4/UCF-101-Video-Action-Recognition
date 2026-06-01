# UCF-101 Video Action Recognition
## MobileNetV2 + LSTM | 101 Action Classes

A deep learning pipeline that classifies human actions in videos
using frame-level feature extraction + temporal sequence modeling.

## Architecture
- Frame Extraction → 8 frames sampled per video
- MobileNetV2 (ImageNet pretrained) → spatial features per frame
- TimeDistributed Dense → frame-wise projection
- LSTM (128 units) → temporal pattern learning
- Dense + Softmax → 101 class prediction

## Dataset
UCF-101 | 101 action classes | Kaggle

## Tech Stack
Python · TensorFlow · Keras · OpenCV · NumPy · Pandas · Matplotlib

## Results
Top-5 predictions with confidence scores per video
