import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.utils import shuffle

# Hàm Load Dữ liệu
def load_data(dirData, image_size=(128, 128)):
    data = []
    labels = []
    
    # Duyệt qua các thư mục và ảnh
    for folder in os.listdir(dirData):
        folder_path = os.path.join(dirData, folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, image_size)
            data.append(img_resized)
            labels.append(folder)  # Lưu nhãn dựa vào tên thư mục
    
    # Chuyển dữ liệu và nhãn thành numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Encode nhãn thành số
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    # Trộn lẫn dữ liệu và nhãn một cách ngẫu nhiên
    data, labels = shuffle(data, labels, random_state=42)  # Trộn dữ liệu và nhãn cùng lúc
    
    return data, labels, le


# Hàm Vẽ Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    plt.figure(figsize=(7,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Hàm Tính Precision, Recall, F1-Score và Accuracy
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    accuracy = np.mean(y_true == y_pred)  # Tính Accuracy
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return precision, recall, f1, accuracy



# Hàm Trích xuất đặc trưng HOG
def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features = []
    for img in images:
        # Chuyển ảnh màu sang grayscale nếu ảnh là RGB
        if len(img.shape) == 3:  # Nếu ảnh có 3 kênh (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển thành ảnh grayscale

        # Trích xuất đặc trưng HOG
        fd = hog(img, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2-Hys')
        hog_features.append(fd)
    return np.array(hog_features)

# Hàm Load mô hình CNN đã huấn luyện
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Hàm Dự đoán với mô hình CNN
def predict_with_cnn(model, X):
    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)  # Lấy nhãn có xác suất cao nhất
    return y_pred, y_prob

# Hàm Trích xuất và Vẽ ROC Curve (nếu cần)
def plot_roc_curve(y_true, y_prob, num_classes):
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # Binarize y_true nếu là one-hot
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    # Tính ROC-AUC cho từng lớp
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
    
    # Vẽ ROC Curve
    plt.figure(figsize=(7,5))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
