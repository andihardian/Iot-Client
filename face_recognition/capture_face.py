import cv2
import os
import sys

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
    print("[INFO] Hadap ke kamera. Tekan 'q' untuk batal.\n")

    count = 0
    while count < sample_count:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample: {count}/{sample_count}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Mendaftarkan: {user_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 122), 2)
        cv2.putText(frame, "Tekan 'q' untuk batal",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow("Smart Door — Capture Wajah", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"\n[✅] {count} foto tersimpan di: {save_dir}")
        print("[INFO] Jalankan: python face_recognition/train_model.py")
    else:
        print("[⚠️] Tidak ada wajah terdeteksi. Coba perbaiki pencahayaan.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python face_recognition/capture_face.py <id> <nama>")
        print("Contoh: python face_recognition/capture_face.py 1 Budi")
        sys.exit(1)
    capture_face(sys.argv[1], " ".join(sys.argv[2:]))