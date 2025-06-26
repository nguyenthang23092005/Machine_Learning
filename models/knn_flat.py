import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
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

# Định nghĩa các tham số tìm kiếm cho KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Thử nghiệm với số láng giềng khác nhau
    'weights': ['uniform', 'distance'],  # Thử nghiệm với các kiểu trọng số khác nhau
    'metric': ['euclidean', 'manhattan', 'chebyshev']  # Các metric khác nhau để tính khoảng cách
}

# Sử dụng GridSearchCV để tìm tham số tối ưu
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# In các thông số tối ưu
print("Tối ưu hóa GridSearchCV hoàn thành!")
print("Tham số tối ưu:", grid_search.best_params_)

# Sử dụng mô hình với các tham số tối ưu đã tìm được
best_knn_model = grid_search.best_estimator_

# Đánh giá mô hình với các tham số tối ưu
evaluate_model(best_knn_model, X_test, y_test)

# Cross-validation trên tập huấn luyện (k-fold, mặc định là 5-fold)
cv_scores = cross_val_score(best_knn_model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

# Đảm bảo thư mục lưu mô hình tồn tại
model_dir = r'D:\GitHub\Machine_Learning\weights'
create_directory(model_dir)

# Lưu mô hình KNN vào file sử dụng joblib
knn_model_path = os.path.join(model_dir, 'knn_model_flat_optimal.joblib')
save_model(best_knn_model, knn_model_path)

print(f'Mô hình KNN tối ưu đã được lưu vào: {knn_model_path}')
