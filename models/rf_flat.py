import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from utils import load_data_flatten, save_model, evaluate_model, create_directory
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Đường dẫn đến thư mục chứa dữ liệu train và test
data_dir = r"D:\data_ML\data_crop"

# Tải dữ liệu
data, label = load_data_flatten(data_dir)

# Kiểm tra kích thước của dữ liệu
print(f"Kích thước dữ liệu: {data.shape}, {label.shape}")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

# Định nghĩa các tham số tìm kiếm cho Random Forest
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

# Đảm bảo thư mục lưu mô hình tồn tại
model_dir = r'D:\GitHub\Machine_Learning\weights'
create_directory(model_dir)

# Lưu mô hình Random Forest vào file sử dụng joblib
rf_model_path = os.path.join(model_dir, 'rf_model_flat_optimal.joblib')
save_model(best_rf_model, rf_model_path)

print(f'Mô hình Random Forest tối ưu đã được lưu vào: {rf_model_path}')
