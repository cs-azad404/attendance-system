# import os
# import cv2

# from .config import FRAME_WIDTH, FRAME_HEIGHT, LBPH_THRESHOLD, today_csv_path
# from .utils import ensure_dirs, load_face_detector, detect_faces, preprocess_face, load_label_map, mark_attendance

# def main():
#     ensure_dirs()

#     # Load model
#     model_path = os.path.join(os.path.dirname(__file__), "..", "data", "model", "lbph_model.xml")
#     if not os.path.exists(model_path):
#         print("[ERROR] Model not found. Train the model first (python src/train_model.py).")
#         return

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(model_path)

#     # Labels
#     label_map = load_label_map()
#     # Convert keys to int if needed
#     label_map = {int(k): v for k, v in label_map.items()}

#     detector = load_face_detector()

#     cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

#     csv_path = today_csv_path()
#     print(f"[INFO] Attendance CSV: {csv_path}")
#     print("[INFO] Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Cannot access webcam.")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detect_faces(gray, detector)

#         for (x, y, w, h) in faces:
#             face_img = preprocess_face(gray, x, y, w, h)
#             label_id, confidence = recognizer.predict(face_img)

#             # LBPH returns distance; lower = better. Accept if <= threshold.
#             if confidence <= LBPH_THRESHOLD and label_id in label_map:
#                 name = label_map[label_id]
#                 print(f"[RECOGNIZED] {name} (user_id={label_id}) | confidence={confidence:.2f}")
#                 mark_attendance(csv_path, label_id, name)
#             else:
#                 print(f"[UNKNOWN] confidence={confidence:.2f}")

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("[INFO] Exiting attendance.")
#             break

#     cap.release()
#     print("[DONE] Attendance session ended.")

# if __name__ == "__main__":
#     main()


import os
import cv2
import time

from .config import FRAME_WIDTH, FRAME_HEIGHT, LBPH_THRESHOLD, today_csv_path
from .utils import ensure_dirs, load_face_detector, detect_faces, preprocess_face, load_label_map, mark_attendance

def main():
    ensure_dirs()

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "..", "data", "model", "lbph_model.xml")
    if not os.path.exists(model_path):
        print("[ERROR] Model not found. Train the model first (python -m src.train_model).")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # Labels
    label_map = load_label_map()
    label_map = {int(k): v for k, v in label_map.items()}

    detector = load_face_detector()

    # Use DirectShow backend for Windows
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    csv_path = today_csv_path()
    print(f"[INFO] Attendance CSV: {csv_path}")
    print("[INFO] Press 'q' in the preview window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        for (x, y, w, h) in faces:
            face_img = preprocess_face(gray, x, y, w, h)
            label_id, confidence = recognizer.predict(face_img)

            if confidence <= LBPH_THRESHOLD and label_id in label_map:
                name = label_map[label_id]
                print(f"[RECOGNIZED] {name} (user_id={label_id}) | confidence={confidence:.2f}")
                mark_attendance(csv_path, label_id, name)
                cv2.putText(frame, f"{name} ({confidence:.1f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                print(f"[UNKNOWN] confidence={confidence:.2f}")
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        # Show live preview window
        cv2.imshow("Attendance - Press q to quit", frame)

        # Exit when 'q' is pressed in the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting attendance.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Attendance session ended.")

if __name__ == "__main__":
    main()
