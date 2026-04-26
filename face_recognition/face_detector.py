import cv2
import os
import sys
import requests
import threading

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
HAAR       = os.path.join(BASE_DIR, '..', 'haarcascade', 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'face_recognition', 'face_model.yml')
API_URL    = 'http://127.0.0.1:8000/api/access-log'

USER_NAMES = {
    1: 'hardi',
}

def send_to_api(identifier, method):
    try:
        requests.post(API_URL, json={
            'identifier': identifier,
            'method': method,
        }, timeout=3)
    except:
        pass

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

    last_sent  = {}   # cegah spam API: {label: frame_count}
    frame_count = 0
    COOLDOWN   = 30   # kirim API tiap 30 frame per orang

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)

            if confidence < 70:
                name       = USER_NAMES.get(label, f'User {label}')
                identifier = f'FACE-user_{label}'
                status     = f'{name} ({confidence:.0f})'
                color      = (0, 255, 0)

                # Kirim ke API dengan cooldown
                if frame_count - last_sent.get(label, -COOLDOWN) >= COOLDOWN:
                    last_sent[label] = frame_count
                    threading.Thread(
                        target=send_to_api,
                        args=(identifier, 'face'),
                        daemon=True
                    ).start()
                    print(f'[GRANTED] {name} - confidence: {confidence:.1f} → API terkirim')
                else:
                    print(f'[GRANTED] {name} - confidence: {confidence:.1f}')

            else:
                status = f'Tidak dikenal ({confidence:.0f})'
                color  = (0, 0, 255)
                print(f'[DENIED] Wajah tidak dikenal - confidence: {confidence:.1f}')

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Smart Door — Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()