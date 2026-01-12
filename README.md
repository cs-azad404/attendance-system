# Attendance System (Console) – Face Recognition with OpenCV LBPH

## Overview
A console-only attendance system using OpenCV's LBPH face recognizer. It captures a per-user face dataset, trains a model, and logs attendance to CSV based on live webcam recognition. No GUI involved.

## ✨ Features 
- Console logs with recognized names and confidence values 
- Per-day CSV attendance (unique mark per person per day) 
- Simple dataset capture and training workflow 
- Uses OpenCV Haar Cascades and LBPH for robustness and easy installation 
- Live preview window during dataset capture and attendance

## Project structure
attendance-system/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ config.py
│  ├─ utils.py
│  ├─ capture_dataset.py
│  ├─ train_model.py
│  ├─ run_attendance.py
├─ data/
│  ├─ dataset/
│  ├─ model/
│  ├─ attendance/



## ⚙️ Requirements
```txt
opencv-contrib-python
numpy
pandas

Install with:
pip install -r requirements.txt




## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows

source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt

## Capture dataset
python -m src.capture_dataset --user-id 1 --name "AZAD" --samples 100
python -m src.capture_dataset --user-id 2 --name "PRIYA" --samples 100
python -m src.capture_dataset --user-id 3 --name "RAHUL" --samples 100

Each user gets their own folder under data/dataset/:
data/dataset/1/   # AZAD
data/dataset/2/   # PRIYA
data/dataset/3/   # RAHUL

labels.json will automatically update to include all users:
{
  "1": "AZAD",
  "2": "PRIYA",
  "3": "RAHUL"
}


## Train model
python -m src.train_model

This rebuilds lbph_model.xml using all users’ images.

## Run attendance
python -m src.run_attendance

Recognizes any enrolled user.
Shows live preview window with bounding boxes and labels.
Logs attendance in data/attendance/attendance_YYYY-MM-DD.csv.

How it works
Uses OpenCV Haar Cascade to detect faces from webcam frames.
Captured grayscale face crops become your dataset.
Trains an LBPHFaceRecognizer and maps label IDs to names.
During attendance, matches faces with thresholds and writes to a daily CSV.

## Example attendance CSV
user_id,name,timestamp
1,AZAD,2026-01-12 21:45:12
2,PRIYA,2026-01-12 21:46:03
3,RAHUL,2026-01-12 21:47:20

Notes
Lighting and camera angle matter. Capture in consistent lighting.
Add multiple users by running capture_dataset.py for each with a unique --user-id.
If recognition is poor, capture more diverse images and retrain.
Press q in the preview window to exit capture or attendance.

License