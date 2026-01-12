import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import cv2
import numpy as np

from .config import DATA_DIR, DATASET_DIR, MODEL_DIR, ATTENDANCE_DIR, HAAR_CASCADE_PATH, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

def load_face_detector():
    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Haar cascade not found at {HAAR_CASCADE_PATH}")
    return cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def detect_faces(gray_frame, detector):
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )
    return faces

def preprocess_face(gray_frame, x, y, w, h):
    face_roi = gray_frame[y:y+h, x:x+w]
    # Normalize size for recognizer stability
    face_resized = cv2.resize(face_roi, (200, 200))
    return face_resized

def dataset_dir_for_user(user_id: int) -> str:
    path = os.path.join(DATASET_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path

def save_label_map(label_map: Dict[int, str]):
    path = os.path.join(MODEL_DIR, "labels.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

def load_label_map() -> Dict[int, str]:
    path = os.path.join(MODEL_DIR, "labels.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def init_attendance_csv(path: str):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["user_id", "name", "timestamp"])
        df.to_csv(path, index=False)

def mark_attendance(path: str, user_id: int, name: str):
    init_attendance_csv(path)
    df = pd.read_csv(path)
    # Only mark once per person per day
    if not ((df["user_id"] == user_id) & (df["name"] == name)).any():
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [user_id, name, ts]
        df.to_csv(path, index=False)

def collect_training_data() -> Tuple[list, list]:
    images = []
    labels = []
    # Each subdirectory in dataset is a user_id
    for user_id_dir in sorted(os.listdir(DATASET_DIR)):
        user_path = os.path.join(DATASET_DIR, user_id_dir)
        if not os.path.isdir(user_path):
            continue
        user_id = int(user_id_dir)
        for fname in os.listdir(user_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(user_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(user_id)
    return images, labels
