import cv2
import os
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR   = os.path.join(BASE_DIR, '..', 'models', 'face_recognition')
MODEL_PATH  = os.path.join(MODEL_DIR, 'face_model.yml')

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, labels = [], []
    for user_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, user_folder)
        if not os.path.isdir(folder_path):
            continue
        try:
            user_id = int(user_folder.split('_')[1])
        except:
            continue
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(user_id)
        print(f'[INFO] User {user_id}: {len(os.listdir(folder_path))} foto dimuat')

    if not faces:
        print('[ERROR] Tidak ada data wajah ditemukan!')
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    print(f'[OK] Model tersimpan di: {MODEL_PATH}')
    print(f'[OK] Total sampel: {len(faces)}')

if __name__ == '__main__':
    print('[INFO] Melatih model...')
    train()
    print('[DONE] Training selesai!')