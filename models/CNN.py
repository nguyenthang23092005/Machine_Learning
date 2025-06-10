import numpy as np
import os
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from utils import load_data, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical  # Import to_categorical

# Đường dẫn đến dữ liệu
train_data = r'D:\data_crop_ML\train_data'
test_data = r'D:\data_crop_ML\test_data'

# Load dữ liệu (train và test)
X_train, y_train, le = load_data(train_data)
X_test, y_test, _ = load_data(test_data)

# Chuyển đổi y_train và y_test thành one-hot encoding
y_train = to_categorical(y_train, num_classes=8)  # 8 là số lớp
y_test = to_categorical(y_test, num_classes=8)

# Xây dựng mô hình CNN
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(8, activation='softmax')  # Số lớp tương ứng với nhãn one-hot
])

# Compile mô hình
model_cnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model_cnn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Dự đoán trên tập test
y_pred = model_cnn.predict(X_test)

# Chuyển đổi dự đoán từ one-hot sang nhãn số nguyên
y_pred = np.argmax(y_pred, axis=1)  # Lấy chỉ số nhãn có xác suất cao nhất

# Chuyển y_test từ dạng one-hot thành số nguyên
y_test_labels = np.argmax(y_test, axis=1)  # Chuyển từ one-hot về số nguyên

# Vẽ Confusion Matrix
plot_confusion_matrix(y_test_labels, y_pred, le.classes_)

# Đánh giá mô hình trên tập test
test_loss, test_accuracy = model_cnn.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Tính các chỉ số khác: Precision, Recall, F1-Score
precision = precision_score(y_test_labels, y_pred, average='macro')
recall = recall_score(y_test_labels, y_pred, average='macro')
f1 = f1_score(y_test_labels, y_pred, average='macro')

print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-Score (macro): {f1:.4f}")

# Lưu mô hình
model_dir = r'D:\GitHub\Machine_Learning\weithts'
model_cnn.save(os.path.join(model_dir, 'cnn_model.h5'))
print("Mô hình đã được lưu.")
