import cv2
import os
import sys
import requests
import threading
import tempfile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
HAAR       = os.path.join(BASE_DIR, '..', 'haarcascade', 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'face_recognition', 'face_model.yml')
API_URL    = 'http://127.0.0.1:8000/api/door/unlock'

TELEGRAM_TOKEN   = '8689794607:AAH-qZm2pPEuJTsodxHoQO8Xi3lpXItcs9I'
TELEGRAM_CHAT_ID = '5438873362'

USER_NAMES = {
    1: 'hardi',
}

def send_telegram_photo(frame, caption):
    try:
        # Simpan frame ke file temporary
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

def send_to_api(identifier, method, frame=None):
    try:
        res  = requests.post(API_URL, json={
            'device_token': 'raspi-token-001',
            'identifier'  : identifier,
            'method'      : method,
        }, timeout=5)
        data = res.json()
        status = data.get('status')

        # Kirim foto ke Telegram
        if frame is not None:
            from datetime import datetime
            waktu = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            if status == 'granted':
                caption = (f"✅ <b>AKSES DITERIMA</b>\n\n"
                          f"🚪 Pintu berhasil dibuka\n"
                          f"📅 Waktu     : {waktu}\n"
                          f"🔑 Identifier: {identifier}\n"
                          f"📡 Metode    : FACE")
            else:
                reason = data.get('reason', 'Tidak diketahui')
                caption = (f"🚨 <b>PERINGATAN SMART DOOR</b>\n\n"
                          f"❌ <b>Akses Ditolak!</b>\n"
                          f"📅 Waktu     : {waktu}\n"
                          f"🔑 Identifier: {identifier}\n"
                          f"📡 Metode    : FACE\n"
                          f"⚠️ Alasan    : {reason}")

            threading.Thread(
                target=send_telegram_photo,
                args=(frame.copy(), caption),
                daemon=True
            ).start()

        return status
    except Exception as e:
        print(f'[API ERROR] {e}')
        return None

def run():
    if not os.path.exists(MODEL_PATH):
        print('[ERROR] Model belum ada! Jalankan train_model.py dulu.')
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(HAAR)
    if face_cascade.empty():
        print('[ERROR] Haarcascade tidak ditemukan!')
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[ERROR] Webcam tidak bisa dibuka!')
        sys.exit(1)

    print('[INFO] Face recognition aktif. Tekan Q untuk keluar.')

    last_sent   = {}
    frame_count = 0
    COOLDOWN    = 60

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
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
            else:
                name       = 'Tidak dikenal'
                identifier = 'FACE-UNKNOWN'
                color      = (0, 0, 255)
                status     = f'Tidak dikenal ({confidence:.0f})'

            # Gambar kotak dulu sebelum capture
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Kirim ke API + foto
            key = identifier
            if frame_count - last_sent.get(key, -COOLDOWN) >= COOLDOWN:
                last_sent[key] = frame_count
                print(f'\n[FACE] Terdeteksi: {name} (confidence: {confidence:.1f})')
                threading.Thread(
                    target=send_to_api,
                    args=(identifier, 'face', frame),
                    daemon=True
                ).start()

        cv2.putText(frame, 'Smart Door — Face Recognition',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 122), 2)
        cv2.imshow('Smart Door — Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()