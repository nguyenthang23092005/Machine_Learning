import os
import cv2
import random
import numpy as np
from mtcnn import MTCNN

# Khởi tạo detector MTCNN
detector = MTCNN()

# Thư mục dữ liệu
train_data = r'D:\data_ML\data_collect\train_data'
test_data = r'D:\data_ML\data_collect\test_data'

# Thư mục lưu ảnh cắt (train_crop_data và test_crop_data)
train_crop_data = r'D:\data_ML\data_crop\train_data'
test_crop_data = r'D:\data_ML\data_crop\test_data'

def crop_data(dirData, save_dir, limit=100):
    data = []
    image_count = len(os.listdir(save_dir))  # Đếm số ảnh đã cắt trong thư mục đích    
    for folder in os.listdir(dirData):
        folder_path = os.path.join(dirData, folder)
        
        # Kiểm tra nếu là thư mục chứa ảnh
        if os.path.isdir(folder_path):
            # Tạo thư mục con tương ứng trong save_dir nếu chưa có
            save_folder = os.path.join(save_dir, folder)
            os.makedirs(save_folder, exist_ok=True)

            # Danh sách ảnh trong thư mục
            images = os.listdir(folder_path)

            if not images:
                print(f"Warning: No images found in {folder_path}. Skipping...")
                continue

            random.shuffle(images)  # Xáo trộn ảnh ngẫu nhiên
            for file in images:
                if image_count >= limit:  # Nếu đã đủ số lượng ảnh, dừng lại
                    return data  # Dừng và trả về dữ liệu nếu đạt giới hạn

                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)

                # Kiểm tra xem ảnh có được đọc đúng không
                if img is None:
                    print(f"Warning: Unable to read image {file}. Skipping...")
                    continue

                # Phát hiện khuôn mặt trong ảnh
                faces = detector.detect_faces(img)

                if not faces:
                    print(f"No faces detected in {file}. Skipping...")
                    continue

                # Lặp qua tất cả các khuôn mặt phát hiện được
                for i, face in enumerate(faces):
                    if image_count >= limit:
                        return data

                    x, y, w, h = face['box']
                    print(f"Crop box: x={x}, y={y}, w={w}, h={h}, img shape={img.shape}")

                    # Đảm bảo crop không vượt ngoài ảnh
                    x, y = max(0, x), max(0, y)
                    face_crop = img[y:y+h, x:x+w]
                    if face_crop.size == 0:
                        print("Warning: Crop ra ảnh rỗng, bỏ qua.")
                        continue

                    face_resized = cv2.resize(face_crop, (128, 128))
                    img_name = f"{file.split('.')[0]}_face_{i}.png"
                    save_path = os.path.join(save_folder, img_name)

                    if cv2.imwrite(save_path, face_resized):
                        print(f"Saved cropped face for {file} as {img_name}")
                        image_count += 1  # Tăng biến đếm
                    else:
                        print(f"Failed to save {img_name}")

                    # Thêm ảnh và nhãn vào data
                    face_input = np.expand_dims(face_resized / 255.0, axis=0)
                    label = labels[folder]
                    data.append((face_input, label))

                    image_count += 1  # Tăng biến đếm ảnh

    print(f"Finished cropping data for {dirData}. Total images processed: {image_count}")
    return data



def ensure_data_sufficiency(train_data_dir, test_data_dir, train_crop_data, test_crop_data, train_limit=100, test_limit=20):
    # Kiểm tra và cắt thêm ảnh cho train data
    for folder in os.listdir(train_data_dir):
        folder_path = os.path.join(train_data_dir, folder)
        if os.path.isdir(folder_path):
            folder_save_path = os.path.join(train_crop_data, folder)
            if not os.path.exists(folder_save_path):
                os.makedirs(folder_save_path, exist_ok=True)  # Tạo thư mục nếu chưa có

            num_train_images = len(os.listdir(folder_save_path))  # Đếm số lượng ảnh trong thư mục crop
            print(f"Folder {folder} trong train có {num_train_images} ảnh đã cắt.")
            if num_train_images < train_limit:
                print(f"Folder {folder} trong train thiếu {train_limit - num_train_images} ảnh. Cắt thêm...")
                crop_data(folder_path, folder_save_path, train_limit - num_train_images)


    # Kiểm tra và cắt thêm ảnh cho test data
    for folder in os.listdir(test_data_dir):
        folder_path = os.path.join(test_data_dir, folder)
        if os.path.isdir(folder_path):
            folder_save_path = os.path.join(test_crop_data, folder)
            if not os.path.exists(folder_save_path):
                os.makedirs(folder_save_path, exist_ok=True)  # Tạo thư mục nếu chưa có

            num_test_images = len(os.listdir(folder_save_path))  # Đếm số lượng ảnh trong thư mục crop
            print(f"Folder {folder} trong test có {num_test_images} ảnh đã cắt.")
            if num_test_images < test_limit:
                print(f"Folder {folder} trong test thiếu {test_limit - num_test_images} ảnh. Cắt thêm...")
                crop_data(folder_path, folder_save_path, test_limit - num_test_images)



# Kiểm tra và đảm bảo đủ dữ liệu
#ensure_data_sufficiency(train_data, test_data, train_crop_data, test_crop_data, train_limit=100, test_limit=20)

crop_data(r"D:\data_ML\data_crop\train_data\Dat",r"D:\data_ML\data_crop\train_data\tttt",5)

