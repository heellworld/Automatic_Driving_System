"""
Configuration file for Autonomous Driving System
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Data paths
DRIVING_LOG_PATH = os.path.join(DATA_DIR, "driving_log.csv")
IMG_DIR = os.path.join(DATA_DIR, "IMG")
LANE_IMAGES_DIR = os.path.join(DATA_DIR, "lane_line_images")
TRAFFIC_SIGN_IMAGES_DIR = os.path.join(DATA_DIR, "traffic_sign_images")

# Model paths
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.h5")
LANE_MODEL_PATH = os.path.join(MODELS_DIR, "lane_model.pkl")

# Training parameters
class TrainingConfig:
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    SHUFFLE = True
    
    # Image preprocessing
    INPUT_SHAPE = (66, 200, 3)
    CROP_TOP = 60
    CROP_BOTTOM = 135
    
    # Data augmentation
    USE_AUGMENTATION = True
    BRIGHTNESS_RANGE = (0.5, 1.5)
    ROTATION_RANGE = 5
    
class SimulatorConfig:
    # Communication
    HOST = "localhost"
    PORT = 4567
    SOCKET_TIMEOUT = 10
    
    # Control parameters
    SPEED_LIMIT = 20
    MAX_STEERING_ANGLE = 1.0
    MIN_THROTTLE = 0.1
    MAX_THROTTLE = 1.0

class LaneDetectionConfig:
    # Image processing
    GAUSSIAN_KERNEL = (11, 11)
    CANNY_LOW_THRESHOLD = 150
    CANNY_HIGH_THRESHOLD = 200
    
    # Bird's eye view
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640
    
    # Lane detection
    LANE_WIDTH = 100
    INTERESTED_LINE_Y_RATIO = 0.9
    
    # Control
    FIXED_THROTTLE = 0.5
    STEERING_SENSITIVITY = 0.01

class WebcamConfig:
    CAMERA_INDEX = 1  # 0 for default camera, 1 for external
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 20
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "webcam_recording.mp4")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(WebcamConfig.OUTPUT_PATH), exist_ok=True) 