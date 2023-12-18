import cv2
import numpy as np


def img_preprocess(images):

    # Chuyển thành màu xám
    gray_img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

    # Áp dụng Gaussian Blur để làm mờ ảnh, giúp giảm nhiễu và chi tiết không cần thiết
    blurred_img = cv2.GaussianBlur(gray_img, (11, 11), 0)

    # Áp dung Canny
    thresh_low = 150
    thresh_high = 200
    canny_img = cv2.Canny(blurred_img, thresh_low, thresh_high)

    return canny_img

def birdview_transform(img):

    IMAGE_H = 480 # chiều cao ảnh đích
    IMAGE_W = 640 # chiều rộng ảnh đích

    # Tạo một mảng numpy chứa tọa độ của các điểm trên hình ảnh gốc.
    # Điểm này quyết định cách hình ảnh sẽ được "cắt" để tạo ra góc nhìn mới.
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])

    # Tạo một mảng numpy chứa tọa độ của các điểm trên hình ảnh đích.
    # Điểm này quyết định hình dạng và vị trí của hình ảnh sau khi biến đổi.
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])

    # Tính toán ma trận biến đổi perspectice từ các điểm src đến dst.
    M = cv2.getPerspectiveTransform(src, dst)

    # Sử dụng ma trận M để áp dụng biến đổi perspectice lên hình ảnh img.
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) 
    return warped_img

def find_left_right_points(image, draw=False):

    im_height, im_width = image.shape[:2]

    # Vạch kẻ sử dụng để xác định tâm đường
    interested_line_y = int(im_height * 0.9)
    if draw is not None:
        cv2.line(draw, (0, interested_line_y),
                 (im_width, interested_line_y), (0, 0, 255), 2)
    interested_line = image[interested_line_y, :]

    # Xác định điểm bên trái và bên phải
    left_point = -1
    right_point = -1
    lane_width = 100
    center = im_width // 2

    # Tìm điểm bên trái và bên phải bằng cách duyệt từ tâm ra
    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    # Dự đoán điểm bên phải khi chỉ nhìn thấy điểm bên trái
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width

    # Dự đoán điểm bên trái khi chỉ thấy điểm bên phải
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    # Vẽ hai điểm trái / phải lên ảnh
    if draw is not None:
        if left_point != -1:
            draw = cv2.circle(
                draw, (left_point, interested_line_y), 7, (255, 255, 0), -1)
        if right_point != -1:
            draw = cv2.circle(
                draw, (right_point, interested_line_y), 7, (0, 255, 0), -1)

    return left_point, right_point

def calculate_control_signal(img, draw=None):

    # Xử lý ảnh đầu vào
    img_lines = img_preprocess(img)

    #Biến đổi góc nhìn birdview
    img_birdview = birdview_transform(img_lines)

    draw[:, :] = birdview_transform(draw)

    # Tìm kiếm điểm trái phải
    left_point, right_point = find_left_right_points(img_birdview, draw=draw)

    # Tính toán góc lái và tốc độ
    # Tốc độ được cố định ở mức 50% tốc độ tối đa
    # Tính toán tốc độ từ góc quay
    throttle = 0.5
    steering_angle = 0
    im_center = img.shape[1] // 2

    if left_point != -1 and right_point != -1:

        # Tính độ lệch
        center_point = (right_point + left_point) // 2
        center_diff =  im_center - center_point

        # Tính góc lái
        steering_angle = - float(center_diff * 0.01)

    return throttle, steering_angle
