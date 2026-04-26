import requests
import time
import os
import threading
import cv2
from dotenv import load_dotenv

load_dotenv('config.env')
API_URL      = os.getenv('API_URL', 'http://127.0.0.1:8000/api/door/unlock')
DEVICE_TOKEN = os.getenv('DEVICE_TOKEN', 'raspi-token-001')
SIMULATE     = os.getenv('SIMULATE', 'true').lower() == 'true'

# ── GPIO ──────────────────────────────────────────
class GPIOSimulator:
    BCM  = 'BCM'
    OUT  = 'OUT'
    HIGH = 1
    LOW  = 0
    def setmode(self, m):   print(f"[GPIO] Mode: {m}")
    def setup(self, p, m):  print(f"[GPIO] Pin {p} → {m}")
    def output(self, p, s): print(f"[GPIO] Pin {p} → {'LOW  ✅ RELAY ON  (Pintu TERBUKA)' if s == 0 else 'HIGH 🔒 RELAY OFF (Pintu TERKUNCI)'}")
    def cleanup(self):      print("[GPIO] Cleanup selesai")

GPIO      = GPIOSimulator() if SIMULATE else __import__('RPi.GPIO', fromlist=['GPIO'])
RELAY_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)

# ── Face Recognition Setup ────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
HAAR       = os.path.join(BASE_DIR, 'haarcascade', 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'face_recognition', 'face_model.yml')

USER_NAMES = {
    1: 'hardi',
}

# ── API ───────────────────────────────────────────
def check_access(identifier, method='rfid'):
    print(f"\n[API] Kirim request → {identifier} ({method})")
    try:
        res  = requests.post(API_URL, json={
            'device_token': DEVICE_TOKEN,
            'identifier':   identifier,
            'method':       method,
        }, timeout=5)
        data = res.json()
        if data.get('status') == 'granted':
            print(f"[✅ GRANTED] {data['message']}")
            buka_pintu()
        else:
            print(f"[❌ DENIED]  {data['message']}")
            tolak_akses()
        return data.get('status') == 'granted'
    except requests.exceptions.ConnectionError:
        print("[ERROR] Laravel tidak jalan! Jalankan: php artisan serve")
    except Exception as e:
        print(f"[ERROR] {e}")
    return False

# ── GPIO Actions ──────────────────────────────────
def buka_pintu():
    print("\n[🚪] ══════════════════════════")
    print("[🚪]   P I N T U  T E R B U K A")
    print("[🚪] ══════════════════════════")
    GPIO.output(RELAY_PIN, GPIO.LOW)
    time.sleep(3)
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    print("[🔒] Pintu terkunci kembali\n")

def tolak_akses():
    print("\n[🚫] ══════════════════════════")
    print("[🚫]   A K S E S  D I T O L A K")
    print("[🚫] ══════════════════════════\n")

# ── Face Recognition Thread ───────────────────────
def face_recognition_thread():
    if not os.path.exists(MODEL_PATH):
        print("[FACE] Model belum ada, face recognition dinonaktifkan.")
        return

    recognizer   = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR)

    if face_cascade.empty():
        print("[FACE] Haarcascade tidak ditemukan!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FACE] Webcam tidak bisa dibuka!")
        return

    print("[FACE] Webcam aktif — face recognition berjalan.")

    last_sent   = {}
    frame_count = 0
    COOLDOWN    = 60  # ~2 detik @ 30fps, cegah spam API

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi           = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence  = recognizer.predict(face_roi)

            if confidence < 70:
                name   = USER_NAMES.get(label, f'User {label}')
                color  = (0, 255, 0)
                status = f'{name} ({confidence:.0f})'

                if frame_count - last_sent.get(label, -COOLDOWN) >= COOLDOWN:
                    last_sent[label] = frame_count
                    print(f"\n[FACE] Wajah dikenali: {name} (confidence: {confidence:.1f})")
                    threading.Thread(
                        target=check_access,
                        args=(f'FACE-user_{label}', 'face'),
                        daemon=True
                    ).start()
            else:
                color  = (0, 0, 255)
                status = f'Tidak dikenal ({confidence:.0f})'

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, "Smart Door — Face Recognition",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 122), 2)
        cv2.imshow('Smart Door — Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── Manual Input (RFID / PIN) ─────────────────────
def manual_input_thread():
    print("\n" + "="*45)
    print("  SMART DOOR — Laptop Mode")
    print("="*45)
    print("  Ketik UID kartu RFID lalu Enter")
    print("  Contoh PIN  : PIN-1234")
    print("  Contoh RFID : RFID-ABC123")
    print("  Ketik 'exit' untuk keluar")
    print("="*45)

    while True:
        try:
            uid = input("\n[SCAN] UID / PIN > ").strip()
        except EOFError:
            break
        if not uid:
            continue
        if uid.lower() == 'exit':
            os._exit(0)
        if uid.upper().startswith("PIN"):
            check_access(uid, method="pin")
        elif uid.upper().startswith("FACE"):
            check_access(uid, method="face")
        else:
            check_access(uid, method="rfid")

# ── Main ──────────────────────────────────────────
if __name__ == "__main__":
    # Jalankan face recognition di background
    t = threading.Thread(target=face_recognition_thread, daemon=True)
    t.start()

    # Manual input di main thread
    manual_input_thread()

    GPIO.cleanup()