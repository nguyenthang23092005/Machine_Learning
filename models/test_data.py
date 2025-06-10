import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data  # Hàm load_data mà bạn đã viết trước đó

# Đường dẫn tới dữ liệu
test_data_path = r'D:\data_crop_ML\train_data'

# Load dữ liệu và trộn
X, y, le = load_data(test_data_path)

# Lấy một ảnh ngẫu nhiên từ dữ liệu đã trộn
index = np.random.randint(0, len(X))  # Lấy chỉ số ngẫu nhiên
sample_image = X[index]
sample_label_idx = y[index]  # Lấy nhãn số nguyên tương ứng
sample_label = le.inverse_transform([sample_label_idx])  # Chuyển nhãn số nguyên về nhãn gốc

# Hiển thị ảnh mẫu
cv2.imshow("Sample Image", sample_image)
cv2.waitKey(0)  # Chờ phím nhấn
cv2.destroyAllWindows()

# In ra nhãn mẫu
print(f"Sample Label: {sample_label[0]}")
