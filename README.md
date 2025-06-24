# Defect Detection and Classification App

This Streamlit app performs defect detection and classification on uploaded images using Facebook's DETR (DEtection TRansformer) model and zero-shot classification with BART. It is designed to identify specific types of defects and provide recommended actions based on the classification.

## Features

- Object detection using DETR (facebook/detr-resnet-50)
- Zero-shot classification using BART (facebook/bart-large-mnli)
- Predefined defect types:
  - Particle contamination
  - Scratch
  - Pattern defect
  - Resist bubble
  - Etch anomaly
- Action recommendations based on defect type
- Visual display of detected regions with bounding boxes
- Log and tabular output of detection results
