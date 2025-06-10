import cv2
import os

# Dict one-hot encoding (dùng để ánh xạ nhãn đến nhãn one-hot)
dict_onehot = {
    'Thang': [1, 0, 0, 0, 0, 0, 0, 0],
    'Su': [0, 1, 0, 0, 0, 0, 0, 0],
    'Nhung': [0, 0, 1, 0, 0, 0, 0, 0],
    'Tuyen': [0, 0, 0, 1, 0, 0, 0, 0],
    'Vu': [0, 0, 0, 0, 1, 0, 0, 0],
    'Dat': [0, 0, 0, 0, 0, 1, 0, 0],
    'Huy': [0, 0, 0, 0, 0, 0, 1, 0],
    'Unknown': [0, 0, 0, 0, 0, 0, 0, 1]
}

# Đường dẫn đến thư mục chứa dữ liệu test
test_data_path = r'D:\data_crop_ML\test_data'

# Chọn một thư mục ngẫu nhiên (tương ứng với nhãn)
folder_name = 'Su'  
folder_path = os.path.join(test_data_path, folder_name)

# Lấy một ảnh mẫu trong thư mục test
sample_image = os.listdir(folder_path)[0]  # Lấy ảnh đầu tiên
sample_image_path = os.path.join(folder_path, sample_image)

# Đọc và hiển thị ảnh mẫu
sample_img = cv2.imread(sample_image_path)
cv2.imshow("Sample Test Image", sample_img)
cv2.waitKey(0)  # Chờ phím nhấn
cv2.destroyAllWindows()

# In ra nhãn one-hot tương ứng dựa trên tên thư mục
label = dict_onehot[folder_name]  # Lấy nhãn từ tên thư mục
print(f"Sample Label for {folder_name}: {label}")
