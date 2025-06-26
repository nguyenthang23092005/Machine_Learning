import cv2
import numpy as np
from mtcnn import MTCNN
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load pre-trained KNN model
knn_model = joblib.load(r"D:\GitHub\Machine_Learning\weights\knn_model_flat_optimal.joblib")  # Change the path as needed

# Labels
labels = ['Thang', 'Su', 'Nhung', 'Tuyen', 'Vu', 'Dat', 'Huy']

# Initialize MTCNN for face detection
detector = MTCNN()

# Read the image from file
image_path = r"C:\Users\Nguyen Van Thang\Pictures\Camera Roll\WIN_20250626_12_57_21_Pro.jpg"  # Change the path as needed
image = cv2.imread(image_path)

# Convert the image from BGR to RGB (MTCNN requires RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using MTCNN
faces = detector.detect_faces(image_rgb)

# Similarity threshold to classify 'Unknown'
similarity_threshold = 0.

if faces:
    for face in faces:
        x, y, w, h = face['box']

        # Crop the face from the image
        face_crop = image[y:y+h, x:x+w]

        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        # Resize the face to a fixed size (128x128)
        face_resized = cv2.resize(face_crop, (128, 128))

        # Flatten the image (flatten the face into a 1D array)
        face_flattened = face_resized.flatten()

        # Predict with the pre-trained KNN model
        knn_label_idx = knn_model.predict([face_flattened])[0]
        knn_label = labels[knn_label_idx]

        # Calculate cosine similarity between the flattened face features and the stored training data features
        all_face_features = knn_model._fit_X  # Access the training data features stored in the KNN model
        distances = cosine_similarity([face_flattened], all_face_features)  # Calculate cosine similarity

        # If the highest similarity is lower than the threshold, label it as 'Unknown'
        if np.max(distances) < similarity_threshold:
            knn_label = "Unknown"  # If the similarity is too low, label as 'Unknown'

        # Print the predicted label
        print(f'Predicted label: {knn_label}')

        # Draw bounding box and label on the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'KNN: {knn_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
else:
    # If no faces are detected, show a message
    cv2.putText(image, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the result
cv2.imshow("Detected Face", image)

# Ensure the output directory exists, if not create it
output_dir = r'D:\GitHub\Machine_Learning\output\knn_flat'
os.makedirs(output_dir, exist_ok=True)

# Save the result to file
output_image_path = os.path.join(output_dir, 'output_image_knn.jpg')
cv2.imwrite(output_image_path, image)
print(f'Result saved to {output_image_path}')

cv2.waitKey(0)
cv2.destroyAllWindows()
