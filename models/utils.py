import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Định nghĩa danh sách các nhãn
labels = {0: 'Thang', 1: 'Su', 2: 'Nhung', 3: 'Tuyen', 4: 'Vu', 5: 'Dat', 6: 'Huy'}

# Hàm để duyệt qua các thư mục con và lấy ảnh cùng nhãn
def load_data_flatten_dir(directory, data_list, labels_list):
    for label, name in labels.items():
        person_dir = os.path.join(directory, name)  # Đường dẫn tới thư mục của từng người
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            # Đọc ảnh và thay đổi kích thước
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))  # Thay đổi kích thước ảnh

            # Làm phẳng ảnh (chuyển thành vector 1D)
            image_flattened = image.flatten()

            # Thêm ảnh đã làm phẳng và nhãn vào danh sách
            data_list.append(image_flattened)
            labels_list.append(label)

# Hàm tải dữ liệu cho train và test
def load_data_flatten(data_dir):
    data = []
    label = []

    # Tải dữ liệu 
    load_data_flatten_dir(data_dir, data, label)

    # Chuyển dữ liệu sang numpy array
    data = np.array(data)
    label = np.array(label)

    return data, label

# Hàm lưu mô hình vào tệp
def save_model(model, model_path):
    joblib.dump(model, model_path)

# Hàm để đánh giá mô hình trên bộ kiểm tra
def evaluate_model(model, X_test, y_test):
    # Dự đoán từ mô hình
    y_pred = model.predict(X_test)

    # Tính toán Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # In báo cáo phân loại (classification report)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(labels.values())))

    # Vẽ confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    # Tính precision, recall, f1-score
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Hàm để vẽ confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(labels.values()), yticklabels=list(labels.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Hàm để tạo thư mục nếu chưa tồn tại
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
