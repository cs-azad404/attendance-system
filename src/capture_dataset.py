# import argparse
# import os
# import cv2
# from datetime import datetime

# from .config import CAPTURE_SAMPLES_DEFAULT, FRAME_WIDTH, FRAME_HEIGHT
# from .utils import ensure_dirs, load_face_detector, detect_faces, preprocess_face, dataset_dir_for_user, save_label_map

# def main():
#     parser = argparse.ArgumentParser(description="Capture dataset images for a user from webcam.")
#     parser.add_argument("--user-id", type=int, required=True, help="Numeric user ID.")
#     parser.add_argument("--name", type=str, required=True, help="Person's display name.")
#     parser.add_argument("--samples", type=int, default=CAPTURE_SAMPLES_DEFAULT, help="Number of samples to capture.")
#     args = parser.parse_args()

#     ensure_dirs()
#     detector = load_face_detector()

#     # Maintain or update labels.json
#     label_map_path_exists = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "model", "labels.json"))
#     if label_map_path_exists:
#         from .utils import load_label_map
#         label_map = load_label_map()
#     else:
#         label_map = {}

#     label_map[str(args.user_id)] = args.name
#     save_label_map({int(k): v for k, v in label_map.items()})

#     cap = cv2.VideoCapture(1)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

#     save_dir = dataset_dir_for_user(args.user_id)
#     count = 0

#     print(f"[INFO] Capturing {args.samples} samples for user_id={args.user_id}, name={args.name}. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Cannot access webcam.")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detect_faces(gray, detector)

#         if len(faces) > 0:
#             # Use first detected face
#             x, y, w, h = faces[0]
#             face_img = preprocess_face(gray, x, y, w, h)
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#             fname = os.path.join(save_dir, f"{args.user_id}_{timestamp}.png")
#             cv2.imwrite(fname, face_img)
#             count += 1
#             print(f"[CAPTURED] {count}/{args.samples} -> {fname}")

#         if count >= args.samples:
#             print("[INFO] Capture complete.")
#             break

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("[INFO] Capture interrupted by user.")
#             break

#     cap.release()
#     print("[DONE] Dataset capture finished.")

# if __name__ == "__main__":
#     main()
import argparse
import os
import cv2
from datetime import datetime

from .config import CAPTURE_SAMPLES_DEFAULT, FRAME_WIDTH, FRAME_HEIGHT
from .utils import ensure_dirs, load_face_detector, detect_faces, preprocess_face, dataset_dir_for_user, save_label_map

def main():
    parser = argparse.ArgumentParser(description="Capture dataset images for a user from webcam.")
    parser.add_argument("--user-id", type=int, required=True, help="Numeric user ID.")
    parser.add_argument("--name", type=str, required=True, help="Person's display name.")
    parser.add_argument("--samples", type=int, default=CAPTURE_SAMPLES_DEFAULT, help="Number of samples to capture.")
    args = parser.parse_args()

    ensure_dirs()
    detector = load_face_detector()

    # Maintain or update labels.json
    label_map_path_exists = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "model", "labels.json"))
    if label_map_path_exists:
        from .utils import load_label_map
        label_map = load_label_map()
    else:
        label_map = {}

    label_map[str(args.user_id)] = args.name
    save_label_map({int(k): v for k, v in label_map.items()})

    # Use DirectShow backend for Windows
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # 👈 you said your camera is index 1, not 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    save_dir = dataset_dir_for_user(args.user_id)
    count = 0

    print(f"[INFO] Capturing {args.samples} samples for user_id={args.user_id}, name={args.name}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        for (x, y, w, h) in faces:
            face_img = preprocess_face(gray, x, y, w, h)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(save_dir, f"{args.user_id}_{timestamp}.png")
            cv2.imwrite(fname, face_img)
            count += 1
            print(f"[CAPTURED] {count}/{args.samples} -> {fname}")

            # Draw rectangle and label on preview
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, f"Sample {count}/{args.samples}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Show live preview window
        cv2.imshow("Dataset Capture - Press q to quit", frame)

        if count >= args.samples:
            print("[INFO] Capture complete.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Capture interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Dataset capture finished.")

if __name__ == "__main__":
    main()
