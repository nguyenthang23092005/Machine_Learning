import cv2
import os
import random
import numpy as np
import imgaug.augmenters as iaa
import shutil

# Các kỹ thuật augmentation sử dụng imgaug
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Lật ảnh theo chiều ngang với xác suất 50%
    iaa.Affine(rotate=(-25, 25)),  # Xoay ảnh ngẫu nhiên trong khoảng -25 đến 25 độ
    iaa.GaussianBlur(sigma=(0, 3.0)),  # Thêm hiệu ứng làm mờ Gaussian
    iaa.Multiply((0.8, 1.2)),  # Thay đổi độ sáng ngẫu nhiên
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Thêm nhiễu Gaussian
    iaa.Crop(percent=(0, 0.1)),  # Cắt ảnh ngẫu nhiên từ 0% đến 10% của ảnh
])

def augment_random_images(input_dir, output_dir, limit=100):
    """
    Hàm chỉnh sửa ngẫu nhiên các ảnh trong thư mục con để làm phong phú dữ liệu.
    Lựa chọn ngẫu nhiên một số ảnh và áp dụng augmentation.

    Args:
    - input_dir (str): Đường dẫn đến thư mục chứa ảnh gốc.
    - output_dir (str): Đường dẫn đến thư mục lưu ảnh đã augment.
    - limit (int): Giới hạn số lượng ảnh sẽ được chỉnh sửa.
    """
    # Lặp qua tất cả các thư mục con trong thư mục input_dir
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        
        if os.path.isdir(folder_path):
            # Tạo thư mục con tương ứng trong output_dir nếu chưa có
            save_folder = os.path.join(output_dir, folder)
            os.makedirs(save_folder, exist_ok=True)

            # Lấy danh sách ảnh trong thư mục con
            image_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
            random.shuffle(image_files)  # Xáo trộn các ảnh ngẫu nhiên

            image_count = 0  # Biến đếm số ảnh đã xử lý
            augmented_files = random.sample(image_files, min(limit, len(image_files)))  # Chọn ngẫu nhiên ảnh để augment

            # Lặp qua tất cả ảnh trong thư mục con
            for file in image_files:
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)

                # Nếu ảnh được chọn để augment
                if file in augmented_files:
                    augmented_image = seq(image=img)
                    # Lưu ảnh đã augment và ghi đè lên ảnh gốc
                    save_path = os.path.join(save_folder, file)
                    cv2.imwrite(save_path, augmented_image)
                else:
                    # Nếu không phải ảnh augment, sao chép ảnh gốc vào thư mục đích
                    save_path = os.path.join(save_folder, file)
                    shutil.copy(img_path, save_path)

                image_count += 1  # Tăng biến đếm ảnh

                # Dừng khi đã đủ số lượng ảnh
                if image_count >= len(image_files):
                    break

# Đường dẫn đến thư mục ảnh gốc và thư mục lưu ảnh đã chỉnh sửa
input_train_dir = r'D:\data_ML\data_crop\train_data'
output_train_dir = r'D:\data_ML\data_edit\train_data'

input_test_dir = r'D:\data_ML\data_crop\test_data'
output_test_dir = r'D:\data_ML\data_edit\test_data'

# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Gọi hàm để thực hiện chỉnh sửa ngẫu nhiên cho ảnh và ghi đè
augment_random_images(input_train_dir, output_train_dir, limit=100)
augment_random_images(input_test_dir, output_test_dir, limit=100)
