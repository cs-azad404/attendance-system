import os
import sys
import cv2
import numpy as np   # <-- add this

from .config import LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y
from .utils import ensure_dirs, collect_training_data

def main():
    ensure_dirs()
    images, labels = collect_training_data()

    if len(images) == 0:
        print("[ERROR] No training images found in data/dataset/. Add data first.")
        sys.exit(1)

    # Convert labels list to numpy array of integers
    labels = np.array(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y
    )
    recognizer.train(images, labels)

    model_path = os.path.join(os.path.dirname(__file__), "..", "data", "model", "lbph_model.xml")
    recognizer.save(model_path)
    print(f"[DONE] Model trained and saved to: {model_path}")

if __name__ == "__main__":
    main()
