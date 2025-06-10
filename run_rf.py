import cv2
import numpy as np
from mtcnn import MTCNN
import joblib
from skimage.feature import hog

# Tải mô hình Random Forest (RF) đã huấn luyện sẵn
rf_model = joblib.load(r'D:\GitHub\Machine_Learning\weithts\rf_model.pkl')

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

    for face in faces:
        x, y, w, h = face['box']

        # Cắt khuôn mặt từ ảnh
        face_crop = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (128, 128))  # Resize khuôn mặt về kích thước phù hợp

        # Chuyển đổi ảnh màu sang ảnh xám (grayscale)
        face_resized_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Trích xuất đặc trưng HOG
        hog_features = hog(face_resized_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features = np.reshape(hog_features, (1, -1))  # Chuyển thành vector 1 chiều

        # Dự đoán với mô hình Random Forest (RF)
        rf_label_idx = rf_model.predict(hog_features)[0]
        rf_label = labels[rf_label_idx]

        # Vẽ khung và nhãn lên khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'RF: {rf_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hiển thị kết quả trên camera
    cv2.imshow("Camera - Nhấn 'q' để thoát", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
