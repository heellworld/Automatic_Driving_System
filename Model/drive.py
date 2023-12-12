import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model  # Make sure to import from tensorflow.keras
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__)  # Flask application
speed_limit = 20

def img_preprocess(img):
    # Preprocess the image as required by your model
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    # Send control command to the Unity app
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
        # Predict the steering angle
        steering_angle = float(model.predict(image))
    except Exception as e:
        print(e)
        steering_angle = 0

    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

if __name__ == '__main__':
    # Load the machine learning model
    model = load_model("model.h5")
    app = socketio.Middleware(sio, app)
    # Start the Socket.IO server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
