# Employee Face Recognition System

## Overview
A Python-based face recognition system that identifies registered employees using a webcam and records recognition details with timestamps.

## Technologies Used
- **Python** – Core programming language
- **InsightFace** – Face recognition library
- **Buffalo_L** – Pretrained face analysis model
- **SCRFD** – Face detection
- **ArcFace + ResNet50** – Face recognition
- **OpenCV** – Webcam and image processing
- **NumPy** – Embedding comparison
- **JSON** – Store recognition records

## Working Process
1. Load registered employee photos from `known_faces/`.
2. Extract 512-dimensional face embeddings.
3. Capture live faces through the webcam.
4. Detect faces using SCRFD.
5. Generate embeddings using ArcFace.
6. Compare embeddings using cosine similarity.
7. Identify employees or mark them as unknown.
8. Save recognition details and timestamps in JSON.

```

## Installation

```bash
pip install insightface opencv-python onnxruntime numpy
```

## Run

```bash
python face.py
```

Press `Q` to exit.

## Current Features
- Employee face recognition
- Unknown face detection
- Similarity-based matching
- Timestamp-based JSON logging
- Duplicate recognition prevention per session

## Future Enhancements
- Automatic check-in/check-out
- Lunch and break tracking
- PostgreSQL integration
- Attendance dashboard
- Working-hours calculation
