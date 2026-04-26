import cv2
import numpy as np
import os
import json

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH  = os.path.join(MODEL_DIR, 'face_model.yml')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')

def load_dataset():
    faces  = []
    labels = []
    label_map = {}  # {user_id_int: "user_name_string"}

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Folder dataset tidak ada: {DATASET_DIR}")
        return faces, labels, label_map

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Nama folder: user_1, user_2, dst
        try:
            user_id = int(folder_name.split('_')[1])
        except (IndexError, ValueError):
            print(f"[SKIP] Folder tidak valid: {folder_name}")
            continue

        label_map[user_id] = folder_name
        image_count = 0

        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(folder_path, img_file)
            img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(user_id)
            image_count += 1

        print(f"[INFO] User {user_id} ({folder_name}): {image_count} foto")

    return faces, labels, label_map

def train():
    print("\n[INFO] Memuat dataset...")
    faces, labels, label_map = load_dataset()

    if len(faces) == 0:
        print("[ERROR] Tidak ada data wajah! Jalankan capture_face.py dulu.")
        return

    print(f"[INFO] Total: {len(faces)} foto dari {len(label_map)} user")
    print("[INFO] Melatih model LBPH...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, 'w') as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    print(f"[✅] Model tersimpan: {MODEL_PATH}")
    print(f"[✅] Labels tersimpan: {LABELS_PATH}")
    print("\n[INFO] Training selesai! Sekarang bisa jalankan face_detector.py")

if __name__ == "__main__":
    train()