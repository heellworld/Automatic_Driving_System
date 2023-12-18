import cv2

# Khởi tạo webcam
cap = cv2.VideoCapture(1)  # 0 là ID mặc định của webcam tích hợp đầu tiên

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    raise IOError("Không thể mở webcam")

try:
    while True:
        # Đọc một frame từ webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị frame
        cv2.imshow('Webcam', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Giải phóng tài nguyên và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()
