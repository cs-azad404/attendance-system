import os
from datetime import datetime

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
MODEL_DIR = os.path.join(DATA_DIR, "model")
ATTENDANCE_DIR = os.path.join(DATA_DIR, "attendance")

# Haar cascade (OpenCV provides default path)
import cv2
HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

# Capture/training parameters
FACE_SCALE_FACTOR = 1.2
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (100, 100)

CAPTURE_SAMPLES_DEFAULT = 100
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# LBPH parameters (tweak if needed)
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_THRESHOLD = 70.0  # Lower is stricter; prediction returns confidence (distance). Use <= threshold.

def today_csv_path():
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
