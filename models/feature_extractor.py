import torch
import os
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

# Định nghĩa FeatureExtractor
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
        for param in self.facenet.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.facenet(x)

# Khởi tạo mô hình FeatureExtractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)

# Thiết lập transform chuẩn hóa ảnh
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo VGGFace2
])

# Định nghĩa nhãn của các lớp (labels)
labels = {0: 'Thang', 1: 'Su', 2: 'Nhung', 3: 'Tuyen', 4: 'Vu', 5: 'Dat', 6: 'Huy'}

# Đường dẫn tới thư mục chứa ảnh huấn luyện và kiểm tra
data_dir = r"D:\data_ML\data_crop"

# Hàm để duyệt qua các thư mục con và lấy ảnh cùng nhãn
def load_data_from_directory(directory, data_list, labels_list):
    for label, name in labels.items():
        person_dir = os.path.join(directory, name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            # Đọc ảnh và thay đổi kích thước
            image = cv2.imread(image_path)
            image = cv2.resize(image, (160, 160))  # Thay đổi kích thước ảnh cho phù hợp với InceptionResnetV1

            # Chuyển từ BGR sang RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Thêm ảnh và nhãn vào danh sách
            data_list.append(image)
            labels_list.append(label)

# Khởi tạo danh sách chứa dữ liệu và nhãn
data = []
label = []

# Tải dữ liệu 
load_data_from_directory(data_dir, data, label)

# Kiểm tra kích thước của dữ liệu
print(f"Kích thước dữ liệu: {len(data)}, {len(label)}")

# Trích xuất đặc trưng từ ảnh huấn luyện và kiểm tra
data_torch = torch.stack([transform(img) for img in data]).to(device)

# Trích xuất đặc trưng
with torch.no_grad():
    data_features = feature_extractor(data_torch)

# Chuyển đổi đặc trưng thành numpy array
data_features = data_features.cpu().numpy()

# Đảm bảo thư mục weights tồn tại
weights_dir = r'D:\GitHub\Machine_Learning\weights'
os.makedirs(weights_dir, exist_ok=True)  # Tạo thư mục nếu chưa có

# Lưu đặc trưng vào thư mục weights
np.save(os.path.join(weights_dir, 'data_features.npy'), data_features)

print("Đặc trưng đã được lưu vào 'weights/data_features.npy'.")
