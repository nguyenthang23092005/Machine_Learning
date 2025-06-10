import cv2
import os
import numpy as np
import random
from mtcnn import MTCNN

# Khởi tạo detector MTCNN
detector = MTCNN()

# Thư mục dữ liệu
train_data = r'D:\data_ML\train_data'
test_data = r'D:\data_ML\test_data'

# Thư mục lưu ảnh cắt (train_crop_data và test_crop_data)
train_crop_data = r'D:\data_crop_ML\train_data'
test_crop_data = r'D:\data_crop_ML\test_data'

# Dict one-hot encoding
dict_onehot = {'Thang':[1,0,0,0,0,0,0,0], 'Su':[0,1,0,0,0,0,0,0], 'Nhung':[0,0,1,0,0,0,0,0], 
               'Tuyen':[0,0,0,1,0,0,0,0], 'Vu':[0,0,0,0,1,0,0,0], 'Dat':[0,0,0,0,0,1,0,0], 
               'Huy':[0,0,0,0,0,0,1,0], 'Unknown':[0,0,0,0,0,0,0,1]}

# Hàm crop và lưu ảnh vào thư mục tương ứng với giới hạn số lượng ảnh cho train và test
def crop_data(dirData, save_dir, limit=100):
    data = []
    image_count = 0  # Biến đếm số ảnh đã lấy

    for folder in os.listdir(dirData):
        folder_path = os.path.join(dirData, folder)
        
        # Kiểm tra nếu là thư mục chứa ảnh
        if os.path.isdir(folder_path):
            # Tạo thư mục con tương ứng trong save_dir nếu chưa có
            save_folder = os.path.join(save_dir, folder)
            os.makedirs(save_folder, exist_ok=True)

            # Danh sách ảnh trong thư mục
            images = os.listdir(folder_path)

            random.shuffle(images)  # Xáo trộn ảnh ngẫu nhiên
            for file in images:
                if image_count >= limit:  # Nếu đã đủ số lượng ảnh, dừng lại
                    break

                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)

                # Phát hiện khuôn mặt trong ảnh
                faces = detector.detect_faces(img)

                # Lặp qua tất cả các khuôn mặt phát hiện được
                for i, face in enumerate(faces):
                    if image_count >= limit:  # Nếu đã đủ số lượng ảnh, dừng lại
                        break

                    x, y, w, h = face['box']

                    # Cắt khuôn mặt từ ảnh
                    face_crop = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_crop, (128, 128))  # Resize khuôn mặt về kích thước phù hợp

                    # Lưu ảnh khuôn mặt đã cắt
                    img_name = f"{file.split('.')[0]}_face_{i}.png"
                    save_path = os.path.join(save_folder, img_name)
                    cv2.imwrite(save_path, face_resized)

                    # Thêm ảnh và nhãn vào data
                    face_input = np.expand_dims(face_resized / 255.0, axis=0)
                    label = dict_onehot[folder]
                    data.append((face_input, label))

                    image_count += 1  # Tăng biến đếm ảnh

    return data


# Hàm kiểm tra và đảm bảo đủ số lượng ảnh cho train và test
def ensure_data_sufficiency(train_data_dir, test_data_dir, train_limit=100, test_limit=20):
    # Kiểm tra và cắt thêm ảnh cho train data
    for folder in os.listdir(train_data_dir):
        folder_path = os.path.join(train_data_dir, folder)
        if os.path.isdir(folder_path):
            num_train_images = len(os.listdir(folder_path))  # Đếm số lượng ảnh trong folder
            if num_train_images < train_limit:
                print(f"Folder {folder} trong train thiếu {train_limit - num_train_images} ảnh. Cắt thêm...")
                crop_data(folder_path, os.path.join(train_crop_data, folder), train_limit - num_train_images)

    # Kiểm tra và cắt thêm ảnh cho test data
    for folder in os.listdir(test_data_dir):
        folder_path = os.path.join(test_data_dir, folder)
        if os.path.isdir(folder_path):
            num_test_images = len(os.listdir(folder_path))  # Đếm số lượng ảnh trong folder
            if num_test_images < test_limit:
                print(f"Folder {folder} trong test thiếu {test_limit - num_test_images} ảnh. Cắt thêm...")
                crop_data(folder_path, os.path.join(test_crop_data, folder), test_limit - num_test_images)


# Kiểm tra và đảm bảo đủ dữ liệu
ensure_data_sufficiency(train_data, test_data, train_limit=100, test_limit=20)
