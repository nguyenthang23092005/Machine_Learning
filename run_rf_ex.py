import cv2
import numpy as np
from mtcnn import MTCNN
import joblib
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import os
from sklearn.ensemble import RandomForestClassifier

# Load pre-trained Random Forest model from joblib
rf_model = joblib.load(r"D:\GitHub\Machine_Learning\weights\rf_model_ex_optimal.joblib")

# Labels
labels = ['Thang', 'Su', 'Nhung', 'Tuyen', 'Vu', 'Dat', 'Huy']

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize InceptionResnetV1 for face feature extraction
feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Image transformation for normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image
image_path = r"C:\Users\Nguyen Van Thang\Pictures\Camera Roll\WIN_20250626_12_57_21_Pro.jpg"
image = cv2.imread(image_path)

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using MTCNN
faces = detector.detect_faces(image_rgb)

# Similarity threshold for handling unknown faces
similarity_threshold = 0.8

if faces:
    for face in faces:
        x, y, w, h = face['box']

        # Crop the face from the image
        face_crop = image[y:y+h, x:x+w]

        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        # Resize the face to 160x160 for feature extraction
        face_resized = cv2.resize(face_crop, (160, 160))

        # Convert the face to a tensor and normalize
        face_tensor = transform(face_resized).unsqueeze(0)

        # Extract face features using InceptionResnetV1
        with torch.no_grad():
            face_features = feature_extractor(face_tensor)

        # Flatten the features for prediction
        face_features = face_features.cpu().numpy().flatten().reshape(1, -1)

        # Predict the label using the Random Forest model
        rf_label_idx = rf_model.predict(face_features)[0]
        rf_label = labels[rf_label_idx]

        # Print the predicted label
        print(f'Predicted label: {rf_label}')  # Print prediction result

        # Draw bounding box and label on the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'RF: {rf_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
else:
    # If no faces are detected, display a message
    cv2.putText(image, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the result
cv2.imshow("Detected Face", image)

# Ensure output directory exists, if not, create it
output_dir = r'D:\GitHub\Machine_Learning\output\rf_ex'
os.makedirs(output_dir, exist_ok=True)

# Save the result to file
output_image_path = os.path.join(output_dir, 'output_image_rf.jpg')
cv2.imwrite(output_image_path, image)
print(f'Result saved to {output_image_path}')

cv2.waitKey(0)
cv2.destroyAllWindows()
