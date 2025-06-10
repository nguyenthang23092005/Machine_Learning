import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, extract_hog_features, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib

# Đường dẫn đến dữ liệu
train_data = r'D:\data_crop_ML\train_data'
test_data = r'D:\data_crop_ML\test_data'

# Load dữ liệu
X_train, y_train, le = load_data(train_data)
X_test, y_test, _ = load_data(test_data)

# Trích xuất đặc trưng HOG cho dữ liệu huấn luyện và kiểm tra
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_hog, y_train)

# Dự đoán với Random Forest
y_pred_rf = rf_model.predict(X_test_hog)

# Vẽ Confusion Matrix
plot_confusion_matrix(y_test, y_pred_rf, le.classes_)

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình học máy (Random Forest, SVM) và trả về các chỉ số đánh giá.
    """
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Tính toán độ chính xác (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Tính các chỉ số khác: Precision, Recall, F1-Score
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    
    # Trả về accuracy và các chỉ số đánh giá
    return accuracy, precision, recall, f1

# Đánh giá mô hình
accuracy, precision, recall, f1 = evaluate_model(rf_model, X_test_hog, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-Score (macro): {f1:.4f}")

# Lưu mô hình Random Forest
model_dir = r'D:\GitHub\Machine_Learning\weithts'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf_model, os.path.join(model_dir, 'rf_model.pkl'))
print("Mô hình Random Forest đã được lưu.")
