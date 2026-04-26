import cv2
import os
import sys
import time

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
HAAR        = os.path.join(BASE_DIR, '..', 'haarcascade', 'haarcascade_frontalface_default.xml')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

def capture_face(user_id, user_name, sample_count=30):
    face_cascade = cv2.CascadeClassifier(HAAR)
    if face_cascade.empty():
        print("[ERROR] Haarcascade tidak ditemukan!")
        sys.exit(1)

    save_dir = os.path.join(DATASET_DIR, f"user_{user_id}")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka!")
        sys.exit(1)

    print(f"\n[INFO] Mendaftarkan wajah: {user_name} (ID: {user_id})")
    print("[INFO] Hadap ke kamera. Mengambil foto otomatis...\n")

    count = 0
    attempts = 0
    max_attempts = 300  # ~30 detik

    while count < sample_count and attempts < max_attempts:
        ret, frame = cap.read()
        attempts += 1

        if not ret:
            print("[WARN] Frame tidak terbaca, skip...")
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_img)
            print(f"  📸 Foto {count}/{sample_count} tersimpan")

        time.sleep(0.1)  # jeda antar frame

    cap.release()

    if count > 0:
        print(f"\n[✅] {count} foto tersimpan di: {save_dir}")
        print("[INFO] Jalankan: python face_recognition/train_model.py")
    else:
        print("[⚠️] Tidak ada wajah terdeteksi.")
        print("[TIPS] Pastikan pencahayaan cukup & wajah menghadap kamera.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python face_recognition/capture_face_headless.py <id> <nama>")
        sys.exit(1)
    capture_face(sys.argv[1], " ".join(sys.argv[2:]))