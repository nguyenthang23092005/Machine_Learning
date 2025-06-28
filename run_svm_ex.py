import cv2
import numpy as np
from mtcnn import MTCNN
import joblib
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import os
from sklearn.svm import SVC  
from sklearn.metrics.pairwise import cosine_similarity

svm_model = joblib.load(r"D:\GitHub\Machine_Learning\weights\svm_model_ex_optimal.joblib")  # Change path if needed

labels = ['Thang', 'Su', 'Nhung', 'Tuyen', 'Vu', 'Dat', 'Huy']
detector = MTCNN()
feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = r"C:\Users\Nguyen Van Thang\Pictures\Camera Roll\WIN_20250626_12_57_21_Pro.jpg"
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(image_rgb)
similarity_threshold = 0.6

if faces:
    for face in faces:
        x, y, w, h = face['box']

        face_crop = image[y:y+h, x:x+w]

        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        face_resized = cv2.resize(face_crop, (160, 160))

        face_tensor = transform(face_resized).unsqueeze(0)

        with torch.no_grad():
            face_features = feature_extractor(face_tensor)

        face_features = face_features.cpu().numpy().flatten().reshape(1, -1)

        svm_label_idx = svm_model.predict(face_features)[0]
        svm_label = labels[svm_label_idx]

        support_vectors = svm_model.support_vectors_  
        distances = cosine_similarity(face_features, support_vectors)  

        if np.max(distances) < similarity_threshold:
            svm_label = "Unknown"

        print(f'Predicted label: {svm_label}')  

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'SVM: {svm_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
else:
    cv2.putText(image, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Detected Face", image)

output_dir = r'D:\GitHub\Machine_Learning\output\svm_ex'
os.makedirs(output_dir, exist_ok=True)

counter = 1
while os.path.exists(output_image_path):
    output_image_path = os.path.join(output_dir, f'output_image_svm_{counter}.jpg')
    counter += 1

cv2.imwrite(output_image_path, image)
print(f'Result saved to {output_image_path}')

cv2.waitKey(0)
cv2.destroyAllWindows()
