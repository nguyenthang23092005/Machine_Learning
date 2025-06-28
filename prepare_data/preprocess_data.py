import os
import cv2
import random
import numpy as np
from mtcnn import MTCNN

# Khởi tạo detector MTCNN
detector = MTCNN()
labels = {0: 'Thang', 1: 'Su', 2: 'Nhung', 3: 'Tuyen', 4: 'Vu', 5: 'Dat', 6: 'Huy'}
# Thư mục dữ liệu
data = r'D:\data_ML\data_collect'

# Thư mục lưu ảnh cắt
crop_data = r'D:\data_ML\data_crop'

def crop_data_function(dirData, save_dir, limit=100):
    data = []
    image_count = len(os.listdir(save_dir))  # Start by counting the images in the save_dir

    for folder in os.listdir(dirData):
        folder_path = os.path.join(dirData, folder)
        
        if os.path.isdir(folder_path):
            save_folder = os.path.join(save_dir, folder)
            os.makedirs(save_folder, exist_ok=True)  # Create save folder if not exist
            images = os.listdir(folder_path)

            if not images:
                print(f"Warning: No images found in {folder_path}. Skipping...")
                continue

            random.shuffle(images)  # Shuffle images for randomness

            for file in images:
                if image_count >= limit:  # Exit when we reach the limit
                    return data

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
                    if image_count >= limit:  # Exit when limit is reached
                        return data

                    x, y, w, h = face['box']
                    print(f"Crop box: x={x}, y={y}, w={w}, h={h}, img shape={img.shape}")

                    # Đảm bảo crop không vượt ngoài ảnh
                    x, y = max(0, x), max(0, y)
                    face_crop = img[y:y+h, x:x+w]
                    if face_crop.size == 0:
                        print("Warning: Crop ra ảnh rỗng, bỏ qua.")
                        continue

                    face_resized = cv2.resize(face_crop, (128, 128))  # Resize the face to 128x128
                    img_name = f"{file.split('.')[0]}_face_{i}.png"
                    save_path = os.path.join(save_folder, img_name)

                    # Lưu ảnh đã crop
                    if cv2.imwrite(save_path, face_resized):
                        print(f"Saved cropped face for {file} as {img_name}")
                        image_count += 1  # Increase image count after saving

                    # Thêm ảnh và nhãn vào data
                    face_input = np.expand_dims(face_resized / 255.0, axis=0)  # Normalize and add batch dimension
                    label = labels.get(folder, "Unknown")  # Default to "Unknown" if label not found
                    data.append((face_input, label))

                    image_count += 1  # Tăng biến đếm ảnh

    print(f"Finished cropping data for {dirData}. Total images processed: {image_count}")
    return data


def ensure_data_sufficiency(data_dir, crop_data, limit=100):
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            folder_save_path = os.path.join(crop_data, folder)
            if not os.path.exists(folder_save_path):
                os.makedirs(folder_save_path, exist_ok=True)  # Tạo thư mục nếu chưa có

            num_train_images = len(os.listdir(folder_save_path))  # Đếm số lượng ảnh trong thư mục crop
            print(f"Folder {folder} có {num_train_images} ảnh đã cắt.")
            if num_train_images < limit:
                print(f"Folder {folder} thiếu {limit - num_train_images} ảnh.")
                crop_data_function(folder_path, folder_save_path, limit - num_train_images)


# Kiểm tra và đảm bảo đủ dữ liệu
ensure_data_sufficiency(data, crop_data, limit=100)
