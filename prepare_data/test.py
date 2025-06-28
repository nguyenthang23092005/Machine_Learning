import cv2
import os


# Đường dẫn đến thư mục chứa dữ liệu test
test_data_path = r'D:\data_ML\data_edit'

# Chọn một thư mục ngẫu nhiên (tương ứng với nhãn)
folder_name = 'Thang'  
folder_path = os.path.join(test_data_path, folder_name)

# Lấy một ảnh mẫu trong thư mục test
sample_image = os.listdir(folder_path)[0]  # Lấy ảnh đầu tiên
sample_image_path = os.path.join(folder_path, sample_image)

# Đọc và hiển thị ảnh mẫu
sample_img = cv2.imread(sample_image_path)
cv2.imshow("Sample Test Image", sample_img)
cv2.waitKey(0)  # Chờ phím nhấn
cv2.destroyAllWindows()

print(f"Sample Label for {folder_name}: {folder_name}")
