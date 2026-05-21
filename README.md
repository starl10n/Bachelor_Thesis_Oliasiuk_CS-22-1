# Bachelor Thesis: Football Player Spatial Position Detection

This repository contains the software implementation of a computer vision method for determining the spatial positions of football players on a 2D field model.

The method processes football video frames, detects objects in the scene, identifies field keypoints, classifies players by team, and maps player positions from image coordinates to a 2D football field model using homography.

## Main Features

- Football player and object detection
- Football field keypoint localization
- Team classification based on visual features
- Coordinate transformation from video frame to 2D field model
- Visualization of detected player positions
- Streamlit-based web interface

## Technologies

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- Supervision
- NumPy
- Transformers
- UMAP
- scikit-learn
- Streamlit

## Method Overview

The proposed method includes three main stages:

1. Detection of football scene objects using YOLO26.
2. Field keypoint localization and player team classification using YOLO26-pose, SigLIP, UMAP, and K-Means.
3. Spatial coordinate transformation using homography to project player positions onto a 2D field model.

## Author

Dmytro Oliasiuk  
Computer Science, CS-22-1  
Khmelnytskyi National University