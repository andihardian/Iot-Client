import requests
import time
import os
import threading
import tempfile
import subprocess
import cv2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('config.env')
API_URL      = os.getenv('API_URL', 'http://127.0.0.1:8000/api/door/unlock')
API_BASE     = API_URL.replace('/api/door/unlock', '')
DEVICE_TOKEN = os.getenv('DEVICE_TOKEN', 'raspi-token-001')
SIMULATE     = os.getenv('SIMULATE', 'true').lower() == 'true'

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN', '8689794607:AAH-qZm2pPEuJTsodxHoQO8Xi3lpXItcs9I')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '5438873362')

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
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HAAR         = os.path.join(BASE_DIR, 'haarcascade', 'haarcascade_frontalface_default.xml')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'face_recognition', 'face_model.yml')
DATASET_DIR  = os.path.join(BASE_DIR, 'face_recognition', 'dataset')
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'face_recognition', 'train_model.py')

USER_NAMES = {
    1: 'hardi',
    2: 'andi',
}

MAX_DENIED_NOTIF  = 3
ACCESS_GRANTED    = threading.Event()
REGISTRATION_LOCK = threading.Lock()
IS_REGISTERING    = threading.Event()

# ── Cek Setting Notifikasi dari Laravel ───────────
def is_notif_enabled(type='granted'):
    try:
        res  = requests.get(f'{API_BASE}/api/settings/notifications', timeout=3)
        data = res.json()
        return data.get(type, True)
    except:
        return True  # default aktif jika gagal cek

# ── Telegram ──────────────────────────────────────
def send_telegram_photo(frame, caption):
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(tmp.name, frame)
        tmp.close()
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
        with open(tmp.name, 'rb') as photo:
            requests.post(url, data={
                'chat_id'    : TELEGRAM_CHAT_ID,
                'caption'    : caption,
                'parse_mode' : 'HTML',
            }, files={'photo': photo}, timeout=10)
        os.unlink(tmp.name)
    except Exception as e:
        print(f'[TELEGRAM ERROR] {e}')

# ── API ───────────────────────────────────────────
def check_access(identifier, method='rfid', frame=None):
    print(f"\n[API] Kirim request → {identifier} ({method})")
    try:
        res  = requests.post(API_URL, json={
            'device_token': DEVICE_TOKEN,
            'identifier':   identifier,
            'method':       method,
        }, timeout=5)
        data   = res.json()
        status = data.get('status')
        waktu  = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

        if status == 'granted':
            print(f"[✅ GRANTED] {data['message']}")
            buka_pintu()
            if frame is not None and is_notif_enabled('granted'):
                caption = (f"✅ <b>AKSES DITERIMA</b>\n\n"
                          f"🚪 Pintu berhasil dibuka\n"
                          f"📅 Waktu     : {waktu}\n"
                          f"🔑 Identifier: {identifier}\n"
                          f"📡 Metode    : {method.upper()}")
                threading.Thread(target=send_telegram_photo,
                                 args=(frame.copy(), caption), daemon=True).start()
            elif frame is not None:
                print("[NOTIF] Notifikasi granted dinonaktifkan.")
        else:
            print(f"[❌ DENIED]  {data['message']}")
            tolak_akses()
            if frame is not None and is_notif_enabled('denied'):
                reason = data.get('reason', data.get('message', 'Tidak diketahui'))
                caption = (f"🚨 <b>PERINGATAN SMART DOOR</b>\n\n"
                          f"❌ <b>Akses Ditolak!</b>\n"
                          f"📅 Waktu     : {waktu}\n"
                          f"🔑 Identifier: {identifier}\n"
                          f"📡 Metode    : {method.upper()}\n"
                          f"⚠️ Alasan    : {reason}")
                threading.Thread(target=send_telegram_photo,
                                 args=(frame.copy(), caption), daemon=True).start()
            elif frame is not None:
                print("[NOTIF] Notifikasi denied dinonaktifkan.")

        return status == 'granted'
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

    last_sent     = {}
    notif_granted = {}
    notif_denied  = 0
    frame_count   = 0
    COOLDOWN      = 60
    last_mtime    = os.path.getmtime(MODEL_PATH)

    while True:
        if IS_REGISTERING.is_set():
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Reload model otomatis jika ada update
        try:
            mtime = os.path.getmtime(MODEL_PATH)
            if mtime != last_mtime:
                last_mtime = mtime
                recognizer.read(MODEL_PATH)
                print("[FACE] Model di-reload otomatis!")
        except Exception:
            pass

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi          = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)

            if confidence < 70:
                name       = USER_NAMES.get(label, f'User {label}')
                identifier = f'FACE-user_{label}'
                color      = (0, 255, 0)
                status     = f'{name} ({confidence:.0f})'

                if frame_count - last_sent.get(identifier, -COOLDOWN) >= COOLDOWN:
                    last_sent[identifier] = frame_count

                    if not notif_granted.get(label, False):
                        notif_granted[label] = True
                        notif_denied = MAX_DENIED_NOTIF
                        ACCESS_GRANTED.set()
                        print(f"\n[FACE] Terdeteksi: {name} (confidence: {confidence:.1f}) → notif dikirim")
                        threading.Thread(
                            target=check_access,
                            args=(identifier, 'face', frame.copy()),
                            daemon=True
                        ).start()
                    else:
                        print(f"\n[FACE] {name} sudah granted — notif tidak dikirim lagi.")

            else:
                name       = 'Tidak dikenal'
                identifier = 'FACE-UNKNOWN'
                color      = (0, 0, 255)
                status     = f'Tidak dikenal ({confidence:.0f})'

                if frame_count - last_sent.get(identifier, -COOLDOWN) >= COOLDOWN:
                    last_sent[identifier] = frame_count

                    if ACCESS_GRANTED.is_set():
                        print(f"\n[FACE] Tidak dikenal — diabaikan (sudah ada akses granted).")
                    elif notif_denied < MAX_DENIED_NOTIF:
                        notif_denied += 1
                        print(f"\n[FACE] Tidak dikenal — notif {notif_denied}/{MAX_DENIED_NOTIF}")
                        threading.Thread(
                            target=check_access,
                            args=(identifier, 'face', frame.copy()),
                            daemon=True
                        ).start()
                    else:
                        print(f"\n[FACE] Tidak dikenal — notif sudah maksimal ({MAX_DENIED_NOTIF}x), diabaikan.")

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

# ── Face Registration Thread ──────────────────────
def face_registration_thread():
    print("[REG] Face registration thread aktif.")

    while True:
        time.sleep(5)
        try:
            res          = requests.get(f'{API_BASE}/api/face-requests/pending', timeout=3)
            pending_list = res.json()

            if not pending_list:
                continue

            for req in pending_list:
                req_id  = req['id']
                user_id = req['user_id']
                name    = req['name']

                print(f"\n[REG] Request pendaftaran: {name} (user_id: {user_id})")

                IS_REGISTERING.set()
                print("[REG] Face recognition dijeda...")
                time.sleep(1)

                with REGISTRATION_LOCK:
                    try:
                        requests.post(f'{API_BASE}/api/face-requests/{req_id}/processing', timeout=3)

                        save_dir = os.path.join(DATASET_DIR, f'user_{user_id}')
                        os.makedirs(save_dir, exist_ok=True)

                        for f in os.listdir(save_dir):
                            os.remove(os.path.join(save_dir, f))

                        face_cascade = cv2.CascadeClassifier(HAAR)
                        cap          = cv2.VideoCapture(0)

                        if not cap.isOpened():
                            print("[REG] Webcam tidak bisa dibuka!")
                            requests.post(f'{API_BASE}/api/face-requests/{req_id}/failed', timeout=3)
                            continue

                        print(f"[REG] Mengambil 30 foto untuk {name}...")
                        count         = 0
                        attempts      = 0
                        no_face_count = 0
                        cancelled     = False

                        while count < 30 and attempts < 500:
                            if attempts % 20 == 0:
                                try:
                                    cr = requests.get(
                                        f'{API_BASE}/api/face-requests/{req_id}/cancelled',
                                        timeout=2
                                    )
                                    if cr.json().get('cancelled'):
                                        print("[REG] Dibatalkan oleh user!")
                                        cancelled = True
                                        break
                                except:
                                    pass

                            ret, frame = cap.read()
                            attempts  += 1
                            if not ret:
                                continue

                            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                            if len(faces) == 0:
                                no_face_count += 1
                                if no_face_count % 30 == 0:
                                    print("[REG] ⚠️ Tidak ada wajah terdeteksi")
                                time.sleep(0.1)
                                continue

                            no_face_count = 0

                            for (x, y, w, h) in faces:
                                if w < 80 or h < 80:
                                    print("[REG] ⚠️ Wajah terlalu kecil/jauh")
                                    continue

                                count   += 1
                                face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                                cv2.imwrite(os.path.join(save_dir, f'{count}.jpg'), face_img)
                                print(f"[REG] 📸 Foto {count}/30 tersimpan")

                                try:
                                    requests.post(
                                        f'{API_BASE}/api/face-requests/{req_id}/progress',
                                        json={'progress': count},
                                        timeout=2
                                    )
                                except:
                                    pass

                            time.sleep(0.1)

                        cap.release()

                        if cancelled:
                            continue

                        if count < 10:
                            print(f"[REG] Gagal — hanya {count} foto terkumpul")
                            requests.post(f'{API_BASE}/api/face-requests/{req_id}/failed', timeout=3)
                            continue

                        print("[REG] Training model...")
                        subprocess.run(['python', TRAIN_SCRIPT], check=True)

                        USER_NAMES[user_id] = name
                        print(f"[REG] USER_NAMES diupdate: {USER_NAMES}")

                        requests.post(f'{API_BASE}/api/face-requests/{req_id}/done', timeout=3)
                        print(f"[REG] ✅ {name} berhasil didaftarkan!")

                    except Exception as e:
                        print(f"[REG] Error: {e}")
                        try:
                            requests.post(f'{API_BASE}/api/face-requests/{req_id}/failed', timeout=3)
                        except:
                            pass

                IS_REGISTERING.clear()
                ACCESS_GRANTED.clear()
                print("[REG] Face recognition aktif kembali.")

        except Exception as e:
            print(f'[REG ERROR] {e}')
            IS_REGISTERING.clear()

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
    t1 = threading.Thread(target=face_recognition_thread, daemon=True)
    t1.start()

    t2 = threading.Thread(target=face_registration_thread, daemon=True)
    t2.start()

    manual_input_thread()

    GPIO.cleanup()