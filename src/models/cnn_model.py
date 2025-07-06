"""
CNN Model for Autonomous Driving - Deep Learning Approach
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import logging

class CNNSteeringModel:
    """CNN model for steering angle prediction"""
    
    def __init__(self, input_shape=(66, 200, 3)):
        self.input_shape = input_shape
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def build_model(self, learning_rate=0.001):
        """Build CNN architecture inspired by NVIDIA's End-to-End Learning"""
        self.model = models.Sequential([
            # Normalization layer
            layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=self.input_shape),
            
            # Convolutional layers
            layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
            layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
            layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Dropout for regularization
            layers.Dropout(0.5),
            
            # Flatten
            layers.Flatten(),
            
            # Fully connected layers
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='relu'),
            
            # Output layer (steering angle)
            layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.logger.info("CNN model built successfully")
        return self.model
    
    def get_callbacks(self, model_path, patience=3):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_generator, validation_generator, epochs=10, 
              model_save_path='model.h5'):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        callbacks = self.get_callbacks(model_save_path)
        
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info(f"Training completed. Model saved to {model_save_path}")
        return history
    
    def evaluate(self, test_generator):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        loss, mae = self.model.evaluate(test_generator, verbose=1)
        self.logger.info(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
        return loss, mae
    
    def predict(self, image):
        """Predict steering angle for a single image"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image)
        return prediction[0][0]  # Return scalar steering angle
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def save_model(self, model_path):
        """Save current model"""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet."
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        summary = mystdout.getvalue()
        
        return summary 