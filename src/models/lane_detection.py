"""
Lane Detection Model - Computer Vision Approach
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional

class LaneDetectionModel:
    """Computer Vision based lane detection for autonomous driving"""
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            'gaussian_kernel': (11, 11),
            'canny_low': 150,
            'canny_high': 200,
            'image_height': 480,
            'image_width': 640,
            'lane_width': 100,
            'interested_line_y_ratio': 0.9,
            'fixed_throttle': 0.5,
            'steering_sensitivity': 0.01
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for lane detection
        
        Args:
            image: Input RGB image
            
        Returns:
            Processed binary image with detected edges
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.config['gaussian_kernel'], 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred, 
            self.config['canny_low'], 
            self.config['canny_high']
        )
        
        return edges
    
    def get_bird_eye_view(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perspective transformation to get bird's eye view
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (warped_image, transformation_matrix)
        """
        height, width = image.shape[:2]
        
        # Define source points (trapezoid in front view)
        src_points = np.float32([
            [0, height],                    # Bottom left
            [width, height],                # Bottom right  
            [0, height * 0.4],             # Top left
            [width, height * 0.4]          # Top right
        ])
        
        # Define destination points (rectangle in bird's eye view)
        dst_points = np.float32([
            [240, height],                  # Bottom left
            [width - 240, height],          # Bottom right
            [-160, 0],                      # Top left
            [width + 160, 0]               # Top right
        ])
        
        # Calculate perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped, M
    
    def find_lane_points(self, binary_image: np.ndarray, 
                        draw_image: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Find left and right lane points
        
        Args:
            binary_image: Binary image with detected edges
            draw_image: Optional image for visualization
            
        Returns:
            Tuple of (left_point, right_point) x-coordinates
        """
        height, width = binary_image.shape[:2]
        
        # Define line of interest (near bottom of image)
        interested_line_y = int(height * self.config['interested_line_y_ratio'])
        
        # Draw reference line for visualization
        if draw_image is not None:
            cv2.line(draw_image, (0, interested_line_y), 
                    (width, interested_line_y), (0, 0, 255), 2)
        
        # Extract the line of interest
        interested_line = binary_image[interested_line_y, :]
        
        # Initialize lane points
        left_point = -1
        right_point = -1
        center = width // 2
        
        # Search for left lane point (from center to left)
        for x in range(center, 0, -1):
            if interested_line[x] > 0:
                left_point = x
                break
        
        # Search for right lane point (from center to right)  
        for x in range(center + 1, width):
            if interested_line[x] > 0:
                right_point = x
                break
        
        # Predict missing lane points based on lane width
        lane_width = self.config['lane_width']
        
        if left_point != -1 and right_point == -1:
            # Only left lane detected, predict right lane
            right_point = left_point + lane_width
        elif right_point != -1 and left_point == -1:
            # Only right lane detected, predict left lane
            left_point = right_point - lane_width
        
        # Draw detected points for visualization
        if draw_image is not None:
            if left_point != -1:
                cv2.circle(draw_image, (left_point, interested_line_y), 
                          7, (255, 255, 0), -1)
            if right_point != -1:
                cv2.circle(draw_image, (right_point, interested_line_y), 
                          7, (0, 255, 0), -1)
        
        return left_point, right_point
    
    def calculate_control_signals(self, image: np.ndarray, 
                                draw_image: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Calculate throttle and steering control signals
        
        Args:
            image: Input image from camera
            draw_image: Optional image for visualization
            
        Returns:
            Tuple of (throttle, steering_angle)
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get bird's eye view
        bird_view, _ = self.get_bird_eye_view(processed_image)
        
        # Apply bird's eye view to visualization image if provided
        if draw_image is not None:
            draw_image[:, :] = self.get_bird_eye_view(draw_image)[0]
        
        # Find lane points
        left_point, right_point = self.find_lane_points(bird_view, draw_image)
        
        # Calculate control signals
        throttle = self.config['fixed_throttle']
        steering_angle = 0.0
        
        image_center = image.shape[1] // 2
        
        if left_point != -1 and right_point != -1:
            # Both lanes detected
            lane_center = (left_point + right_point) // 2
            center_deviation = image_center - lane_center
            
            # Calculate steering angle based on deviation
            steering_angle = -float(center_deviation * self.config['steering_sensitivity'])
            
            self.logger.debug(f"Lane center: {lane_center}, Deviation: {center_deviation}")
        else:
            self.logger.warning("Unable to detect both lane lines")
        
        # Clamp steering angle to reasonable range
        steering_angle = np.clip(steering_angle, -1.0, 1.0)
        
        return throttle, steering_angle
    
    def visualize_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Create visualization of lane detection
        
        Args:
            image: Input image
            
        Returns:
            Annotated image showing detection results
        """
        # Create copy for visualization
        vis_image = image.copy()
        
        # Calculate control signals with visualization
        throttle, steering_angle = self.calculate_control_signals(image, vis_image)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, f'Throttle: {throttle:.2f}', 
                   (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f'Steering: {steering_angle:.3f}', 
                   (10, 60), font, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def update_config(self, new_config: dict):
        """Update configuration parameters"""
        self.config.update(new_config)
        self.logger.info("Configuration updated")
    
    def get_config(self) -> dict:
        """Get current configuration"""
        return self.config.copy() 