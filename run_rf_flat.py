import cv2
import os
import numpy as np
from mtcnn import MTCNN
import joblib

# Tải mô hình Random Forest đã huấn luyện sẵn
rf_model = joblib.load(r"D:\GitHub\Machine_Learning\weights\rf_model_flat_optimal.joblib")  # Thay đổi đường dẫn nếu cần

# Danh sách nhãn
labels = ['Thang', 'Su', 'Nhung', 'Tuyen', 'Vu', 'Dat', 'Huy']

# Khởi tạo MTCNN để phát hiện khuôn mặt
detector = MTCNN()

# Đọc ảnh từ file
image_path = r"C:\Users\Nguyen Van Thang\Pictures\Camera Roll\WIN_20250626_12_57_21_Pro.jpg"  # Thay đổi đường dẫn ảnh của bạn
image = cv2.imread(image_path)

# Chuyển ảnh từ BGR sang RGB (MTCNN yêu cầu ảnh RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Phát hiện khuôn mặt bằng MTCNN
faces = detector.detect_faces(image_rgb)

# Ngưỡng độ tương đồng để phân loại "Unknown"
similarity_threshold = 0.8

if faces:
    for face in faces:
        x, y, w, h = face['box']

        # Cắt khuôn mặt từ ảnh
        face_crop = image[y:y+h, x:x+w]

        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        face_resized = cv2.resize(face_crop, (128, 128))  # Resize khuôn mặt về kích thước phù hợp
        face_flattened = face_resized.flatten()  # Làm phẳng ảnh khuôn mặt

        # Dự đoán với mô hình Random Forest
        rf_label_idx = rf_model.predict([face_flattened])[0]
        rf_label = labels[rf_label_idx]

        # In ra nhãn dự đoán
        print(f'Predicted label: {rf_label}')  # In kết quả dự đoán

        # Vẽ khung và nhãn lên khuôn mặt
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'RF: {rf_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
else:
    # Nếu không phát hiện khuôn mặt, hiển thị thông báo
    cv2.putText(image, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Hiển thị kết quả
cv2.imshow("Detected Face", image)

# Đảm bảo thư mục tồn tại, nếu không tạo thư mục
output_dir = r'D:\GitHub\Machine_Learning\output\rf_flat'
os.makedirs(output_dir, exist_ok=True)

# Lưu kết quả vào file
output_image_path = os.path.join(output_dir, 'output_image_rf.jpg')
cv2.imwrite(output_image_path, image)
print(f'Result saved to {output_image_path}')

cv2.waitKey(0)
cv2.destroyAllWindows()
