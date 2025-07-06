#!/usr/bin/env python3
"""
Real-time autonomous driving script
Supports both CNN and lane detection approaches with simulator communication
"""
import os
import sys
import argparse
import asyncio
import json
import base64
import time
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import socketio
import eventlet
from flask import Flask

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import SimulatorConfig, WebcamConfig, MODELS_DIR, LOGS_DIR
from src.models.cnn_model import CNNSteeringModel
from src.models.lane_detection import LaneDetectionModel
from src.utils import setup_logging, preprocess_image, draw_steering_info, create_video_writer

class AutonomousDrivingController:
    """Main controller for autonomous driving"""
    
    def __init__(self, approach='cnn', model_path=None, debug=False):
        self.approach = approach
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.cnn_model = None
        self.lane_model = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_steering = 0.0
        
        # Video recording
        self.video_writer = None
        self.record_video = False
        
        # Load models based on approach
        if approach in ['cnn', 'hybrid']:
            self.load_cnn_model(model_path)
        
        if approach in ['lane', 'hybrid']:
            self.load_lane_model()
    
    def load_cnn_model(self, model_path):
        """Load CNN model"""
        try:
            if model_path is None:
                model_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
            
            self.cnn_model = CNNSteeringModel()
            self.cnn_model.load_model(model_path)
            self.logger.info(f"CNN model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load CNN model: {e}")
            if self.approach == 'cnn':
                raise
    
    def load_lane_model(self):
        """Load lane detection model"""
        try:
            self.lane_model = LaneDetectionModel()
            self.logger.info("Lane detection model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize lane model: {e}")
            if self.approach == 'lane':
                raise
    
    def predict_steering(self, image):
        """Predict steering angle based on selected approach"""
        steering_angle = 0.0
        throttle = 0.5
        debug_info = {}
        
        try:
            if self.approach == 'cnn' and self.cnn_model:
                # CNN prediction
                processed_image = preprocess_image(image)
                steering_angle = self.cnn_model.predict(processed_image)
                throttle = 1.0 - abs(steering_angle)  # Slow down on turns
                debug_info['method'] = 'CNN'
                
            elif self.approach == 'lane' and self.lane_model:
                # Lane detection prediction
                throttle, steering_angle = self.lane_model.calculate_control_signals(image)
                debug_info['method'] = 'Lane Detection'
                
            elif self.approach == 'hybrid' and self.cnn_model and self.lane_model:
                # Hybrid approach - combine both methods
                processed_image = preprocess_image(image)
                cnn_steering = self.cnn_model.predict(processed_image)
                lane_throttle, lane_steering = self.lane_model.calculate_control_signals(image)
                
                # Weighted combination (can be tuned)
                steering_angle = 0.7 * cnn_steering + 0.3 * lane_steering
                throttle = lane_throttle
                
                debug_info['method'] = 'Hybrid'
                debug_info['cnn_steering'] = cnn_steering
                debug_info['lane_steering'] = lane_steering
                
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            steering_angle = self.last_steering * 0.8  # Gradual decay
        
        # Apply safety limits
        steering_angle = np.clip(steering_angle, -SimulatorConfig.MAX_STEERING_ANGLE, 
                               SimulatorConfig.MAX_STEERING_ANGLE)
        throttle = np.clip(throttle, SimulatorConfig.MIN_THROTTLE, SimulatorConfig.MAX_THROTTLE)
        
        self.last_steering = steering_angle
        
        return steering_angle, throttle, debug_info

class SimulatorInterface:
    """Interface for simulator communication"""
    
    def __init__(self, controller, record_video=False):
        self.controller = controller
        self.sio = socketio.Server()
        self.app = Flask(__name__)
        self.record_video = record_video
        
        # Setup socket events
        self.setup_socket_events()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
    
    def setup_socket_events(self):
        """Setup socketio event handlers"""
        
        @self.sio.on('connect')
        def connect(sid, environ):
            self.controller.logger.info("Simulator connected")
            self.send_control(0, 0)
        
        @self.sio.on('disconnect')
        def disconnect(sid):
            self.controller.logger.info("Simulator disconnected")
        
        @self.sio.on('telemetry')
        def telemetry(sid, data):
            if data:
                self.process_telemetry(data)
    
    def send_control(self, steering_angle, throttle):
        """Send control commands to simulator"""
        self.sio.emit('steer', data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        })
    
    def process_telemetry(self, data):
        """Process telemetry data from simulator"""
        try:
            # Extract data
            speed = float(data.get('speed', 0))
            image_data = data.get('image', '')
            
            if not image_data:
                return
            
            # Decode image
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = np.asarray(image)
            
            # Predict steering and throttle
            steering_angle, throttle, debug_info = self.controller.predict_steering(image)
            
            # Apply speed limit
            if speed > SimulatorConfig.SPEED_LIMIT:
                throttle = 0.0
            
            # Send control commands
            self.send_control(steering_angle, throttle)
            
            # Logging and debugging
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Log every 30 frames
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                log_msg = f"Frame {self.frame_count}: "
                log_msg += f"Speed={speed:.1f}, Steering={steering_angle:.3f}, "
                log_msg += f"Throttle={throttle:.3f}, FPS={fps:.1f}"
                
                if debug_info:
                    log_msg += f", Method={debug_info.get('method', 'Unknown')}"
                
                self.controller.logger.info(log_msg)
            
            # Video recording
            if self.record_video and self.controller.video_writer:
                # Add debug information to frame
                vis_image = draw_steering_info(image.copy(), steering_angle, speed, throttle)
                self.controller.video_writer.write(cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            self.controller.logger.error(f"Error processing telemetry: {e}")
            self.send_control(0, 0)  # Emergency stop
    
    def run(self, host=SimulatorConfig.HOST, port=SimulatorConfig.PORT):
        """Run the simulator interface"""
        self.controller.logger.info(f"Starting simulator interface on {host}:{port}")
        
        # Initialize video recording if requested
        if self.record_video:
            output_path = os.path.join(LOGS_DIR, f'driving_recording_{int(time.time())}.mp4')
            self.controller.video_writer = create_video_writer(output_path, 20, (640, 480))
            self.controller.logger.info(f"Recording video to {output_path}")
        
        try:
            # Wrap Flask app with socketio middleware
            app = socketio.Middleware(self.sio, self.app)
            
            # Start server
            eventlet.wsgi.server(eventlet.listen((host, port)), app)
            
        except KeyboardInterrupt:
            self.controller.logger.info("Shutdown requested by user")
        except Exception as e:
            self.controller.logger.error(f"Server error: {e}")
        finally:
            if self.controller.video_writer:
                self.controller.video_writer.release()
                self.controller.logger.info("Video recording saved")

class WebcamInterface:
    """Interface for webcam testing"""
    
    def __init__(self, controller, camera_index=None):
        self.controller = controller
        self.camera_index = camera_index or WebcamConfig.CAMERA_INDEX
    
    def run(self, duration=None, save_video=False):
        """Run webcam interface"""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {self.camera_index}")
        
        self.controller.logger.info(f"Starting webcam interface (camera {self.camera_index})")
        
        # Video writer setup
        video_writer = None
        if save_video:
            output_path = WebcamConfig.OUTPUT_PATH
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = create_video_writer(output_path, WebcamConfig.FPS, frame_size)
            self.controller.logger.info(f"Recording video to {output_path}")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict steering
                steering_angle, throttle, debug_info = self.controller.predict_steering(frame)
                
                # Add visualization
                vis_frame = draw_steering_info(frame.copy(), steering_angle, None, throttle)
                
                # Add method info
                if debug_info.get('method'):
                    cv2.putText(vis_frame, f"Method: {debug_info['method']}", 
                               (10, vis_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Calculate and display FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(vis_frame, f'FPS: {fps:.1f}', 
                           (10, vis_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('Autonomous Driving - Webcam Test', vis_frame)
                
                # Save video
                if video_writer:
                    video_writer.write(vis_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Check duration
                if duration and elapsed > duration:
                    break
                    
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            self.controller.logger.info(f"Webcam test completed: {frame_count} frames in {elapsed:.2f}s (avg {fps:.2f} FPS)")

def main():
    parser = argparse.ArgumentParser(description='Real-time autonomous driving')
    parser.add_argument('--approach', choices=['cnn', 'lane', 'hybrid'], default='cnn',
                       help='Driving approach to use')
    parser.add_argument('--model', help='Path to CNN model file')
    parser.add_argument('--interface', choices=['simulator', 'webcam'], default='simulator',
                       help='Interface to use')
    parser.add_argument('--host', default=SimulatorConfig.HOST, help='Simulator host')
    parser.add_argument('--port', type=int, default=SimulatorConfig.PORT, help='Simulator port')
    parser.add_argument('--camera', type=int, help='Camera index for webcam interface')
    parser.add_argument('--duration', type=int, help='Test duration in seconds (webcam only)')
    parser.add_argument('--record', action='store_true', help='Record video')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        os.path.join(LOGS_DIR, 'driving_realtime.log'),
        level=log_level
    )
    
    logger.info("Starting real-time autonomous driving")
    logger.info(f"Arguments: {args}")
    
    try:
        # Initialize controller
        controller = AutonomousDrivingController(
            approach=args.approach,
            model_path=args.model,
            debug=args.debug
        )
        
        # Run interface
        if args.interface == 'simulator':
            interface = SimulatorInterface(controller, record_video=args.record)
            interface.run(host=args.host, port=args.port)
        
        elif args.interface == 'webcam':
            interface = WebcamInterface(controller, camera_index=args.camera)
            interface.run(duration=args.duration, save_video=args.record)
        
        logger.info("Autonomous driving session completed")
        
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    except Exception as e:
        logger.error(f"Session failed: {e}")
        raise

if __name__ == "__main__":
    main() 