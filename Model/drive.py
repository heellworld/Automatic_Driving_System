import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 20

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])

    try:
        steering_angle = float(model.predict(image)[0])
    except Exception as e:
        print(e)
        steering_angle = 0

    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

if __name__ == '__main__':
    model = load_model("model1.h5")
    
    # Khởi tạo Webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam.")
        choice = input("Thử lại? (y/n): ")
        if choice.lower() != 'y':
            exit()

    # Lấy chiều rộng và chiều cao của frame từ webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('D:\\projects\\Automatic_Driving_System\\Model\\output.mp4', fourcc, 20.0, (frame_width, frame_height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Quá trình lưu video hoàn tất.")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)