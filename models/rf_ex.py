import torch
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import joblib
from utils import evaluate_model

# Định nghĩa danh sách các nhãn
labels = {0: 'Thang', 1: 'Su', 2: 'Nhung', 3: 'Tuyen', 4: 'Vu', 5: 'Dat', 6: 'Huy'}

# Đường dẫn đến thư mục chứa các file đặc trưng
weights_dir = r'D:\GitHub\Machine_Learning\weights'

# Đọc đặc trưng đã được lưu từ file .npy
X_features = np.load(os.path.join(weights_dir, 'data_features.npy'))

# Đọc nhãn từ file nếu có
labels_list = []
for label, name in labels.items():
    person_dir = os.path.join(r"D:\data_ML\data_crop", name)
    
    # Gán nhãn cho ảnh huấn luyện
    for image_name in os.listdir(person_dir):
        labels_list.append(label)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_features, labels_list, test_size=0.2, random_state=42)

# Tìm các tham số tối ưu cho mô hình Random Forest sử dụng GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Số lượng cây trong Random Forest
    'max_depth': [None, 10, 20, 30],  # Độ sâu tối đa của cây
    'min_samples_split': [2, 5, 10],  # Số lượng mẫu tối thiểu để chia một node
    'min_samples_leaf': [1, 2, 4],  # Số lượng mẫu tối thiểu trong một lá
    'bootstrap': [True, False]  # Chọn cách tạo bootstrap cho mô hình
}

# Sử dụng GridSearchCV để tìm tham số tối ưu cho Random Forest
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# In các thông số tối ưu
print("Tối ưu hóa GridSearchCV hoàn thành!")
print("Tham số tối ưu:", grid_search.best_params_)

# Sử dụng mô hình với các tham số tối ưu đã tìm được
best_rf_model = grid_search.best_estimator_

# Đánh giá mô hình với các tham số tối ưu
evaluate_model(best_rf_model, X_test, y_test)

# Cross-validation trên tập huấn luyện (k-fold, mặc định là 5-fold)
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

# Lưu mô hình Random Forest tối ưu vào file sử dụng joblib
rf_model_path = os.path.join(weights_dir, 'rf_model_ex_optimal.joblib')
joblib.dump(best_rf_model, rf_model_path)
print(f'Mô hình Random Forest tối ưu đã được lưu vào: {rf_model_path}')
