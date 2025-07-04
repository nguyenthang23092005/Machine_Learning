import cv2
import os

label = "Vu"
save_path = os.path.join(r"D:\data_ML\data_collect", label)
os.makedirs(save_path, exist_ok=True)
cap = cv2.VideoCapture(0)
i = 0

print("Đang khởi động camera. Nhấn 'q' để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình!")
        continue

    i += 1

    # Hiển thị khung hình gốc để người dùng xem
    cv2.imshow("Camera - Nhấn 'q' để thoát", frame)

    if i > 50 and i <= 1050:
        # Resize ảnh về 256*256 trước khi lưu
        resized_frame = cv2.resize(frame, (256, 256))
        img_name = os.path.join(save_path, f"{i-50}.png")
        cv2.imwrite(img_name, resized_frame)
        print(f"Đã lưu ảnh {i-50} tại {img_name} (kích thước 256*256)")

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Thoát chương trình.")
        break

cap.release()
cv2.destroyAllWindows()
