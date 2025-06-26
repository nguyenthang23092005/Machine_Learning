import cv2
import numpy as np
from mtcnn import MTCNN
import joblib
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained KNN model
knn_model = joblib.load(r"D:\GitHub\Machine_Learning\weights\knn_model_ex_optimal.joblib")

# Labels
labels = ['Thang', 'Su', 'Nhung', 'Tuyen', 'Vu', 'Dat', 'Huy']

# Initialize MTCNN face detector
detector = MTCNN()

# Initialize InceptionResnetV1 for feature extraction
feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Image transformation for normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image
image_path = r"C:\Users\Nguyen Van Thang\Pictures\Camera Roll\WIN_20250626_12_57_21_Pro.jpg"
image = cv2.imread(image_path)

# Convert image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using MTCNN
faces = detector.detect_faces(image_rgb)

# Similarity threshold for unknown faces
similarity_threshold = 0.6

if faces:
    for face in faces:
        x, y, w, h = face['box']

        # Crop the face from the image
        face_crop = image[y:y+h, x:x+w]

        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        # Resize face to 160x160 for feature extraction
        face_resized = cv2.resize(face_crop, (160, 160))

        # Convert to tensor and normalize
        face_tensor = transform(face_resized).unsqueeze(0)

        # Extract features using InceptionResnetV1
        with torch.no_grad():
            face_features = feature_extractor(face_tensor)

        # Flatten features for KNN prediction
        face_features = face_features.cpu().numpy().flatten().reshape(1, -1)

        # Predict with KNN
        knn_label_idx = knn_model.predict(face_features)[0]
        knn_label = labels[knn_label_idx]

        # Optionally, calculate cosine similarity for better handling of unknown faces
        all_face_features = knn_model._fit_X  # Access training data features (embeddings)
        distances = cosine_similarity(face_features, all_face_features)

        # If similarity is lower than threshold, label as Unknown
        if np.max(distances) < similarity_threshold:
            knn_label = "Unknown"

        # Display predicted label
        print(f'Predicted label: {knn_label}')

        # Draw bounding box and label on the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'KNN: {knn_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
else:
    # If no face is detected, show a message
    cv2.putText(image, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the output image
cv2.imshow("Detected Face", image)

# Ensure output directory exists, if not, create it
output_dir = r'D:\GitHub\Machine_Learning\output\knn_ex'
os.makedirs(output_dir, exist_ok=True)

# Save the result to file
output_image_path = os.path.join(output_dir, 'output_image_knn.jpg')
cv2.imwrite(output_image_path, image)
print(f'Result saved to {output_image_path}')

cv2.waitKey(0)
cv2.destroyAllWindows()
