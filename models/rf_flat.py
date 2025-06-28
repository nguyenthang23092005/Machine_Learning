import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from utils import load_data_flatten, save_model, evaluate_model, create_directory
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

data_dir = r"D:\data_ML\data_crop"

data, label = load_data_flatten(data_dir)
print(f"Kích thước dữ liệu: {data.shape}, {label.shape}")

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200], 
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'bootstrap': [True, False]  
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Tối ưu hóa GridSearchCV hoàn thành!")
print("Tham số tối ưu:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

evaluate_model(best_rf_model, X_test, y_test)

cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

model_dir = r'D:\GitHub\Machine_Learning\weights'
create_directory(model_dir)

rf_model_path = os.path.join(model_dir, 'rf_model_flat_optimal.joblib')
save_model(best_rf_model, rf_model_path)

print(f'Mô hình Random Forest tối ưu đã được lưu vào: {rf_model_path}')
