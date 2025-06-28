import torch
import os
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
        for param in self.facenet.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.facenet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo VGGFace2
])

labels = {0: 'Thang', 1: 'Su', 2: 'Nhung', 3: 'Tuyen', 4: 'Vu', 5: 'Dat', 6: 'Huy'}
data_dir = r"D:\data_ML\data_crop"

def load_data_from_directory(directory, data_list, labels_list):
    for label, name in labels.items():
        person_dir = os.path.join(directory, name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            image = cv2.imread(image_path)
            image = cv2.resize(image, (160, 160))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data_list.append(image)
            labels_list.append(label)

data = []
label = []

load_data_from_directory(data_dir, data, label)
print(f"Kích thước dữ liệu: {len(data)}, {len(label)}")
data_torch = torch.stack([transform(img) for img in data]).to(device)

with torch.no_grad():
    data_features = feature_extractor(data_torch)

data_features = data_features.cpu().numpy()

weights_dir = r'D:\GitHub\Machine_Learning\weights'
os.makedirs(weights_dir, exist_ok=True) 

np.save(os.path.join(weights_dir, 'data_features.npy'), data_features)

print("Đặc trưng đã được lưu vào 'weights/data_features.npy'.")
