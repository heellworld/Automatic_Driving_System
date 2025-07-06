#!/usr/bin/env python3
"""
Testing script for autonomous driving models
"""
import os
import sys
import argparse
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (TrainingConfig, LaneDetectionConfig, 
                       DATA_DIR, MODELS_DIR, LOGS_DIR)
from src.models.cnn_model import CNNSteeringModel
from src.models.lane_detection import LaneDetectionModel
from src.utils import (setup_logging, load_driving_data, preprocess_image, 
                      calculate_fps, draw_steering_info)

class ModelTester:
    """Test autonomous driving models"""
    
    def __init__(self, logger):
        self.logger = logger
        self.cnn_model = None
        self.lane_model = None
    
    def load_cnn_model(self, model_path):
        """Load CNN model"""
        try:
            self.cnn_model = CNNSteeringModel()
            self.cnn_model.load_model(model_path)
            self.logger.info(f"CNN model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load CNN model: {e}")
            raise
    
    def load_lane_model(self, config=None):
        """Load lane detection model"""
        try:
            self.lane_model = LaneDetectionModel(config)
            self.logger.info("Lane detection model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize lane model: {e}")
            raise
    
    def test_cnn_accuracy(self, test_data_path, img_dir, max_samples=1000):
        """Test CNN model accuracy on test dataset"""
        if self.cnn_model is None:
            raise ValueError("CNN model not loaded")
        
        # Load test data
        test_data = load_driving_data(test_data_path, img_dir)
        if test_data is None:
            raise ValueError("Failed to load test data")
        
        # Limit samples for faster testing
        if len(test_data) > max_samples:
            test_data = test_data.sample(n=max_samples, random_state=42)
        
        predictions = []
        ground_truth = []
        failed_predictions = 0
        
        self.logger.info(f"Testing CNN model on {len(test_data)} samples...")
        
        for idx, row in test_data.iterrows():
            try:
                # Load and preprocess image
                img_path = row['center']
                true_steering = row['steering']
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_image = preprocess_image(image)
                
                # Predict
                predicted_steering = self.cnn_model.predict(processed_image)
                
                predictions.append(predicted_steering)
                ground_truth.append(true_steering)
                
            except Exception as e:
                self.logger.warning(f"Failed to predict for {img_path}: {e}")
                failed_predictions += 1
                continue
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        
        self.logger.info(f"CNN Model Test Results:")
        self.logger.info(f"  Samples tested: {len(predictions)}")
        self.logger.info(f"  Failed predictions: {failed_predictions}")
        self.logger.info(f"  Mean Squared Error: {mse:.6f}")
        self.logger.info(f"  Mean Absolute Error: {mae:.6f}")
        
        # Plot predictions vs ground truth
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(ground_truth, predictions, alpha=0.6)
        plt.plot([-1, 1], [-1, 1], 'r--', lw=2)
        plt.xlabel('Ground Truth Steering')
        plt.ylabel('Predicted Steering')
        plt.title('Predictions vs Ground Truth')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        errors = predictions - ground_truth
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(LOGS_DIR, 'cnn_test_results.png')
        plt.savefig(plot_path)
        self.logger.info(f"Test results plot saved to {plot_path}")
        plt.show()
        
        return mse, mae
    
    def test_real_time_performance(self, test_video_path=None, duration=30):
        """Test real-time performance with webcam or video"""
        if test_video_path:
            cap = cv2.VideoCapture(test_video_path)
            source = f"video: {test_video_path}"
        else:
            cap = cv2.VideoCapture(0)  # Webcam
            source = "webcam"
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open {source}")
        
        self.logger.info(f"Testing real-time performance with {source}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start_time = time.time()
                
                # Test CNN model if available
                if self.cnn_model is not None:
                    try:
                        processed = preprocess_image(frame)
                        cnn_steering = self.cnn_model.predict(processed)
                        
                        # Draw CNN prediction
                        cv2.putText(frame, f'CNN Steering: {cnn_steering:.3f}', 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception as e:
                        self.logger.warning(f"CNN prediction failed: {e}")
                        cnn_steering = 0.0
                
                # Test lane detection model if available
                if self.lane_model is not None:
                    try:
                        vis_frame = frame.copy()
                        throttle, lane_steering = self.lane_model.calculate_control_signals(frame, vis_frame)
                        
                        # Draw lane detection result
                        cv2.putText(frame, f'Lane Steering: {lane_steering:.3f}', 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f'Throttle: {throttle:.2f}', 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    except Exception as e:
                        self.logger.warning(f"Lane detection failed: {e}")
                        lane_steering = 0.0
                
                # Calculate FPS
                frame_time = time.time() - frame_start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                
                cv2.putText(frame, f'FPS: {fps:.1f}', 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('Real-time Test', frame)
                
                frame_count += 1
                
                # Break on 'q' key or after duration
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if time.time() - start_time > duration:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate average FPS
        elapsed_time = time.time() - start_time
        avg_fps = calculate_fps(frame_count, elapsed_time)
        
        self.logger.info(f"Real-time test completed:")
        self.logger.info(f"  Duration: {elapsed_time:.2f} seconds")
        self.logger.info(f"  Frames processed: {frame_count}")
        self.logger.info(f"  Average FPS: {avg_fps:.2f}")
        
        return avg_fps
    
    def compare_models(self, test_images_dir, num_samples=100):
        """Compare CNN and lane detection models"""
        if self.cnn_model is None or self.lane_model is None:
            raise ValueError("Both models must be loaded for comparison")
        
        # Get test images
        image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {test_images_dir}")
        
        # Limit samples
        image_files = image_files[:num_samples]
        
        cnn_predictions = []
        lane_predictions = []
        processing_times_cnn = []
        processing_times_lane = []
        
        self.logger.info(f"Comparing models on {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(test_images_dir, img_file)
            
            try:
                # Load image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # CNN prediction
                start_time = time.time()
                processed = preprocess_image(image)
                cnn_steering = self.cnn_model.predict(processed)
                cnn_time = time.time() - start_time
                
                # Lane detection prediction
                start_time = time.time()
                throttle, lane_steering = self.lane_model.calculate_control_signals(image)
                lane_time = time.time() - start_time
                
                cnn_predictions.append(cnn_steering)
                lane_predictions.append(lane_steering)
                processing_times_cnn.append(cnn_time)
                processing_times_lane.append(lane_time)
                
            except Exception as e:
                self.logger.warning(f"Failed to process {img_file}: {e}")
                continue
        
        # Analysis
        cnn_predictions = np.array(cnn_predictions)
        lane_predictions = np.array(lane_predictions)
        
        self.logger.info("Model Comparison Results:")
        self.logger.info(f"  CNN average processing time: {np.mean(processing_times_cnn)*1000:.2f} ms")
        self.logger.info(f"  Lane detection average processing time: {np.mean(processing_times_lane)*1000:.2f} ms")
        self.logger.info(f"  CNN steering range: [{np.min(cnn_predictions):.3f}, {np.max(cnn_predictions):.3f}]")
        self.logger.info(f"  Lane steering range: [{np.min(lane_predictions):.3f}, {np.max(lane_predictions):.3f}]")
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(cnn_predictions, label='CNN', alpha=0.7)
        plt.plot(lane_predictions, label='Lane Detection', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Steering Angle')
        plt.title('Steering Predictions Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.scatter(cnn_predictions, lane_predictions, alpha=0.6)
        plt.xlabel('CNN Steering')
        plt.ylabel('Lane Detection Steering')
        plt.title('CNN vs Lane Detection')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.hist(processing_times_cnn, alpha=0.7, label='CNN', bins=20)
        plt.hist(processing_times_lane, alpha=0.7, label='Lane Detection', bins=20)
        plt.xlabel('Processing Time (s)')
        plt.ylabel('Frequency')
        plt.title('Processing Time Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        differences = np.abs(cnn_predictions - lane_predictions)
        plt.hist(differences, bins=20, alpha=0.7)
        plt.xlabel('Absolute Difference')
        plt.ylabel('Frequency')
        plt.title('Prediction Differences')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(LOGS_DIR, 'model_comparison.png')
        plt.savefig(plot_path)
        self.logger.info(f"Comparison plot saved to {plot_path}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test autonomous driving models')
    parser.add_argument('--cnn-model', help='Path to CNN model file')
    parser.add_argument('--test-data', help='Path to test driving log CSV')
    parser.add_argument('--test-images', help='Path to test images directory') 
    parser.add_argument('--test-video', help='Path to test video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time test')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum samples for accuracy test')
    parser.add_argument('--compare', action='store_true', help='Compare CNN and lane detection models')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        os.path.join(LOGS_DIR, 'testing.log'),
        level=log_level
    )
    
    logger.info("Starting model testing")
    logger.info(f"Arguments: {args}")
    
    # Initialize tester
    tester = ModelTester(logger)
    
    try:
        # Load models
        if args.cnn_model:
            tester.load_cnn_model(args.cnn_model)
        
        # Always initialize lane detection model
        tester.load_lane_model()
        
        # Run tests
        if args.test_data and args.cnn_model:
            img_dir = os.path.join(os.path.dirname(args.test_data), 'IMG')
            tester.test_cnn_accuracy(args.test_data, img_dir, args.max_samples)
        
        if args.webcam or args.test_video:
            video_path = args.test_video if args.test_video else None
            tester.test_real_time_performance(video_path, args.duration)
        
        if args.compare and args.test_images:
            tester.compare_models(args.test_images)
        
        logger.info("Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    main() 