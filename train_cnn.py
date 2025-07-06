#!/usr/bin/env python3
"""
Training script for CNN-based autonomous driving model
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import TrainingConfig, DATA_DIR, MODELS_DIR, LOGS_DIR
from src.models.cnn_model import CNNSteeringModel
from src.utils import setup_logging, load_driving_data, preprocess_image, augment_image, plot_training_history, save_model_info

class DrivingDataGenerator(Sequence):
    """Data generator for training the CNN model"""
    
    def __init__(self, df, batch_size=32, use_augmentation=True, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, index):
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        # Initialize batch arrays
        batch_images = np.zeros((self.batch_size, 66, 200, 3))
        batch_steering = np.zeros(self.batch_size)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Load image
            img_path = row['center']
            steering = row['steering']
            
            try:
                # Load and preprocess image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Data augmentation
                if self.use_augmentation:
                    image, steering = augment_image(image, steering)
                
                # Preprocess
                processed_image = preprocess_image(image)
                
                batch_images[i] = processed_image
                batch_steering[i] = steering
                
            except Exception as e:
                logging.warning(f"Error loading image {img_path}: {e}")
                # Use zeros for failed images
                batch_images[i] = np.zeros((66, 200, 3))
                batch_steering[i] = 0.0
        
        return batch_images, batch_steering
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_data_generators(csv_path, img_dir, config):
    """Create training and validation data generators"""
    # Load data
    data = load_driving_data(csv_path, img_dir)
    if data is None:
        raise ValueError("Failed to load driving data")
    
    # Filter out zero steering angles to balance dataset
    # Keep all non-zero and random sample of zero steering
    non_zero = data[data['steering'] != 0]
    zero_steering = data[data['steering'] == 0].sample(frac=0.3)  # Keep 30% of zero steering
    balanced_data = pd.concat([non_zero, zero_steering]).reset_index(drop=True)
    
    logging.info(f"Original dataset size: {len(data)}")
    logging.info(f"Balanced dataset size: {len(balanced_data)}")
    
    # Split data
    train_df, val_df = train_test_split(
        balanced_data, 
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=None  # Can't stratify continuous values
    )
    
    # Create generators
    train_generator = DrivingDataGenerator(
        train_df,
        batch_size=config.BATCH_SIZE,
        use_augmentation=config.USE_AUGMENTATION,
        shuffle=config.SHUFFLE
    )
    
    val_generator = DrivingDataGenerator(
        val_df,
        batch_size=config.BATCH_SIZE,
        use_augmentation=False,  # No augmentation for validation
        shuffle=False
    )
    
    logging.info(f"Training samples: {len(train_df)}")
    logging.info(f"Validation samples: {len(val_df)}")
    logging.info(f"Training batches: {len(train_generator)}")
    logging.info(f"Validation batches: {len(val_generator)}")
    
    return train_generator, val_generator

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for autonomous driving')
    parser.add_argument('--data-dir', default=DATA_DIR, help='Data directory path')
    parser.add_argument('--model-name', default='cnn_model.h5', help='Model file name')
    parser.add_argument('--epochs', type=int, default=TrainingConfig.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=TrainingConfig.BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=TrainingConfig.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        os.path.join(LOGS_DIR, 'training.log'), 
        level=log_level
    )
    
    logger.info("Starting CNN model training")
    logger.info(f"Arguments: {args}")
    
    # Update config with command line arguments
    config = TrainingConfig()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.USE_AUGMENTATION = not args.no_augmentation
    
    # Paths
    csv_path = os.path.join(args.data_dir, 'driving_log.csv')
    img_dir = os.path.join(args.data_dir, 'IMG')
    model_path = os.path.join(MODELS_DIR, args.model_name)
    
    # Check if data exists
    if not os.path.exists(csv_path):
        logger.error(f"Driving log not found: {csv_path}")
        return
    
    if not os.path.exists(img_dir):
        logger.error(f"Images directory not found: {img_dir}")
        return
    
    try:
        # Create data generators
        logger.info("Creating data generators...")
        train_generator, val_generator = create_data_generators(csv_path, img_dir, config)
        
        # Create and build model
        logger.info("Building CNN model...")
        model = CNNSteeringModel(input_shape=config.INPUT_SHAPE)
        model.build_model(learning_rate=config.LEARNING_RATE)
        
        # Print model summary
        logger.info("Model architecture:")
        logger.info(model.get_model_summary())
        
        # Train model
        logger.info("Starting training...")
        history = model.train(
            train_generator=train_generator,
            validation_generator=val_generator,
            epochs=config.EPOCHS,
            model_save_path=model_path
        )
        
        # Save model info
        info_path = os.path.join(MODELS_DIR, f"{args.model_name}_info.txt")
        save_model_info(model.model, info_path, config)
        
        # Plot training history
        history_plot_path = os.path.join(LOGS_DIR, 'training_history.png')
        plot_training_history(history, save_path=history_plot_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Model info saved to: {info_path}")
        logger.info(f"Training history plot saved to: {history_plot_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 