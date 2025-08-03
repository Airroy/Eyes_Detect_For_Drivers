# ไฟล์รันหลัก
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# โหลดโมเดลที่เทรนไว้
model = load_model("my_model.keras")

# ตั้งค่าของ mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cv2.namedWindow("Eye Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye Detection", 1280, 720)

# ฟังก์ชันปรับภาพให้คมชัด
def enhance_image(image):
    image = cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
    return image

# ฟังก์ชันตัดภาพดวงตา
def get_eye_image(frame, landmarks, indices, size=(64, 64)):
    h, w, _ = frame.shape
    x1 = int(landmarks[indices[0]].x * w)
    y1 = int(landmarks[indices[0]].y * h)
    x2 = int(landmarks[indices[1]].x * w)
    y2 = int(landmarks[indices[1]].y * h)

    eye_margin = 5
    x_min = max(min(x1, x2) - eye_margin, 0)
    y_min = max(min(y1, y2) - eye_margin, 0)
    x_max = min(max(x1, x2) + eye_margin, w)
    y_max = min(max(y1, y2) + eye_margin, h)

    eye_img = frame[y_min:y_max, x_min:x_max]
    if eye_img.size == 0:
        return None

    eye_img = cv2.resize(eye_img, size)
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)  # ทำเป็นขาวดำ
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)  # แปลงกลับเป็น 3-channel RGB
    eye_img = eye_img / 255.0
    eye_img = eye_img.astype('float32')
    eye_img = eye_img.reshape(1, size[0], size[1], 3)  # (1, 64, 64, 3)
    return eye_img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    enhanced_frame = enhance_image(frame)
    rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye_indices = [33, 133]    # ดวงตาซ้าย
            right_eye_indices = [362, 263]  # ดวงตาขวา

            left_eye_img = get_eye_image(frame, face_landmarks.landmark, left_eye_indices)
            right_eye_img = get_eye_image(frame, face_landmarks.landmark, right_eye_indices)

            if left_eye_img is not None and right_eye_img is not None:
                left_pred = model.predict(left_eye_img, verbose=0)[0][0]
                right_pred = model.predict(right_eye_img, verbose=0)[0][0]

                left_state = "Open" if left_pred > 0.5 else "Closed"
                right_state = "Open" if right_pred > 0.5 else "Closed"

                cv2.putText(frame, f"L: {left_state}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"R: {right_state}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # วาด landmark จุดตา
            for idx in left_eye_indices + right_eye_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    cv2.imshow("Eye Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
