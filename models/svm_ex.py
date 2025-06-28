import os
import numpy as np
import cv2
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from utils import evaluate_model

labels = {0: 'Thang', 1: 'Su', 2: 'Nhung', 3: 'Tuyen', 4: 'Vu', 5: 'Dat', 6: 'Huy'}

weights_dir = r'D:\GitHub\Machine_Learning\weights'

X_features = np.load(os.path.join(weights_dir, 'data_features.npy'))

labels_list = []
for label, name in labels.items():
    person_dir = os.path.join(r"D:\data_ML\data_crop", name)
    
    for image_name in os.listdir(person_dir):
        labels_list.append(label)

X_train, X_test, y_train, y_test = train_test_split(X_features, labels_list, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100],  
    'kernel': ['linear', 'rbf'],  
    'gamma': ['scale', 'auto'],  
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Tối ưu hóa GridSearchCV hoàn thành!")
print("Tham số tối ưu:", grid_search.best_params_)

best_svm_model = grid_search.best_estimator_

evaluate_model(best_svm_model, X_test, y_test)

cv_scores = cross_val_score(best_svm_model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

svm_model_path = os.path.join(weights_dir, 'svm_model_ex_optimal.joblib')
joblib.dump(best_svm_model, svm_model_path)
print(f'Mô hình SVM tối ưu đã được lưu vào: {svm_model_path}')
