import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Tải mô hình CNN đã huấn luyện sẵn
cnn_model = load_model(r'D:\GitHub\Machine_Learning\weithts\cnn_model.h5')

# Danh sách nhãn
labels = ['Thang', 'Su', 'Nhung', 'Tuyen', 'Vu', 'Dat', 'Huy', 'Unknown']

# Khởi tạo MTCNN để phát hiện khuôn mặt
detector = MTCNN()

# Mở video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình từ camera!")
        break

    # Phát hiện khuôn mặt bằng MTCNN
    faces = detector.detect_faces(frame)

    if faces:
        for face in faces:
            x, y, w, h = face['box']

            # Cắt khuôn mặt từ ảnh
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                continue

            face_resized = cv2.resize(face_crop, (128, 128))  # Resize khuôn mặt về kích thước phù hợp
            face_input = np.expand_dims(face_resized / 255.0, axis=0)

            # Dự đoán với mô hình CNN
            cnn_prob = cnn_model.predict(face_input, verbose=0)[0]
            cnn_label_idx = np.argmax(cnn_prob)
            cnn_label = labels[cnn_label_idx]

            # Vẽ khung và nhãn lên khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'CNN: {cnn_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # Nếu không phát hiện khuôn mặt, hiển thị thông báo
        cv2.putText(frame, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị kết quả trên camera
    cv2.imshow("Camera - Nhấn 'q' để thoát", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
