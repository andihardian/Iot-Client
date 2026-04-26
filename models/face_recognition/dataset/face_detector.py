import cv2
import requests
import os
import json
import time
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', 'config.env'))

API_URL              = os.getenv('API_URL', 'http://127.0.0.1:8000/api/door/unlock')
DEVICE_TOKEN         = os.getenv('DEVICE_TOKEN', 'raspi-token-001')
CONFIDENCE_THRESHOLD = 60
COOLDOWN_SECONDS     = 5

HAAR        = os.path.join(BASE_DIR, '..', 'haarcascade', 'haarcascade_frontalface_default.xml')
MODEL_PATH  = os.path.join(BASE_DIR, '..', 'models', 'face_model.yml')
LABELS_PATH = os.path.join(BASE_DIR, '..', 'models', 'labels.json')

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAAR)
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Haarcascade tidak ditemukan: {HAAR}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model belum ada! Jalankan train_model.py dulu.")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.load(MODEL_PATH)

        with open(LABELS_PATH) as f:
            self.labels = json.load(f)

        self.last_sent = 0
        print("[✅] Face Detector siap!")
        print(f"[INFO] User terdaftar : {list(self.labels.values())}")
        print(f"[INFO] Threshold      : {CONFIDENCE_THRESHOLD}")
        print(f"[INFO] Cooldown       : {COOLDOWN_SECONDS}s")

    def send_api(self, identifier):
        now = time.time()
        if now - self.last_sent < COOLDOWN_SECONDS:
            return None
        try:
            res = requests.post(API_URL, json={
                "device_token": DEVICE_TOKEN,
                "identifier":   identifier,
                "method":       "face",
            }, timeout=5)
            self.last_sent = now
            return res.json()
        except requests.exceptions.ConnectionError:
            print("[ERROR] Laravel tidak jalan!")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def draw_ui(self, frame, x, y, w, h, name, conf_pct, is_known):
        color = (0, 220, 80) if is_known else (0, 0, 220)
        label = f"{name}  {conf_pct}%" if is_known else f"Unknown  {conf_pct}%"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-36), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        status = "GRANTED — Pintu Terbuka" if is_known else "DENIED — Akses Ditolak"
        cv2.rectangle(frame, (x, y+h), (x+w, y+h+28), color, -1)
        cv2.putText(frame, status, (x+5, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Webcam tidak bisa dibuka!")
            return

        print("\n[READY] Face Recognition aktif — tekan 'q' untuk keluar\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face_roi       = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label_id, conf = self.recognizer.predict(face_roi)
                is_known       = conf < CONFIDENCE_THRESHOLD
                conf_pct       = max(0, 100 - int(conf))
                user_label     = self.labels.get(str(label_id), "Unknown")
                identifier     = f"FACE-{user_label}" if is_known else "FACE-UNKNOWN"

                self.draw_ui(frame, x, y, w, h, user_label, conf_pct, is_known)

                result = self.send_api(identifier)
                if result:
                    if result.get('status') == 'granted':
                        print(f"[✅ GRANTED] {user_label} — {conf_pct}% confidence")
                    elif result.get('status') == 'denied':
                        print(f"[❌ DENIED]  Wajah tidak dikenal — {conf_pct}%")

            # Header
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 62), (15, 15, 20), -1)
            cv2.putText(frame, "SMART DOOR — Face Recognition",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 122), 2)
            cv2.putText(frame, f"Detected: {len(faces)} face(s) | 'q' to quit",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow("Smart Door — Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n[EXIT] Face Detector selesai")

if __name__ == "__main__":
    try:
        detector = FaceDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except KeyboardInterrupt:
        print("\n[EXIT] Dihentikan")