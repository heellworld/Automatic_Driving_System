"""
Utility functions for Autonomous Driving System
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import logging

def setup_logging(log_file="driving_system.log", level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_driving_data(csv_path, img_dir):
    """Load driving data from CSV file"""
    try:
        data = pd.read_csv(csv_path)
        # Add full path to images
        data['center'] = data['center'].apply(lambda x: os.path.join(img_dir, os.path.basename(x)))
        data['left'] = data['left'].apply(lambda x: os.path.join(img_dir, os.path.basename(x)))
        data['right'] = data['right'].apply(lambda x: os.path.join(img_dir, os.path.basename(x)))
        return data
    except Exception as e:
        logging.error(f"Error loading driving data: {e}")
        return None

def preprocess_image(image, input_shape=(66, 200, 3), crop_top=60, crop_bottom=135):
    """Preprocess image for CNN model"""
    # Crop region of interest
    image = image[crop_top:crop_bottom, :, :]
    
    # Convert RGB to YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Resize
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    
    # Normalize
    image = image / 255.0
    
    return image

def augment_image(image, steering_angle, brightness_range=(0.5, 1.5)):
    """Augment image for data augmentation"""
    # Random brightness
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Random flip
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    
    return image, steering_angle

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot training & validation accuracy if available
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    else:
        ax2.set_title('No Accuracy Data')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
    
    plt.show()

def calculate_fps(frame_count, elapsed_time):
    """Calculate FPS"""
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0

def draw_steering_info(image, steering_angle, speed=None, throttle=None):
    """Draw steering information on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw steering angle
    cv2.putText(image, f'Steering: {steering_angle:.3f}', 
                (10, 30), font, 0.7, (0, 255, 0), 2)
    
    if speed is not None:
        cv2.putText(image, f'Speed: {speed:.1f}', 
                    (10, 60), font, 0.7, (0, 255, 0), 2)
    
    if throttle is not None:
        cv2.putText(image, f'Throttle: {throttle:.3f}', 
                    (10, 90), font, 0.7, (0, 255, 0), 2)
    
    return image

def save_model_info(model, filepath, training_config=None):
    """Save model information to text file"""
    with open(filepath, 'w') as f:
        # Model summary
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Training configuration
        if training_config:
            f.write("\n\n=== Training Configuration ===\n")
            for attr, value in vars(training_config).items():
                if not attr.startswith('_'):
                    f.write(f"{attr}: {value}\n")

def check_image_integrity(image_path):
    """Check if image file is valid"""
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False

def create_video_writer(output_path, fps, frame_size):
    """Create OpenCV VideoWriter"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def resize_image_keep_aspect(image, target_width, target_height):
    """Resize image while keeping aspect ratio"""
    h, w = image.shape[:2]
    aspect = w / h
    
    if aspect > target_width / target_height:
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        new_height = target_height
        new_width = int(target_height * aspect)
    
    resized = cv2.resize(image, (new_width, new_height))
    
    # Add padding if needed
    if new_width < target_width or new_height < target_height:
        top = (target_height - new_height) // 2
        bottom = target_height - new_height - top
        left = (target_width - new_width) // 2
        right = target_width - new_width - left
        
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return resized 