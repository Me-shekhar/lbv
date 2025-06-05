import numpy as np
import pandas as pd
import joblib
import os
from data_processor import DataProcessor
from model_trainer import ModelTrainer

class PredictionModel:
    """Main prediction model class for LBV predictions"""
    
    def __init__(self):
        self.model = None
        self.data_processor = DataProcessor()
        self.trainer = ModelTrainer()
        self.is_trained = False
        self.hydrocarbon_stats = {}
    
    def train(self, df):
        """
        Train the prediction model
        
        Args:
            df: Preprocessed DataFrame with LBV data
        """
        try:
            # Check if pre-trained model exists
            if os.path.exists('lbv_model.pkl') and os.path.exists('preprocessors.pkl'):
                print("Loading pre-trained model...")
                if self._load_pretrained_model():
                    self._compute_hydrocarbon_stats(df)
                    return True
            
            print("Training new model...")
            print(f"Dataset size: {len(df)} samples")
            print(f"Unique hydrocarbons: {df['Hydrocarbon'].nunique()}")
            
            # Train the model
            self.model, training_scores = self.trainer.train_models(df)
            self.data_processor = self.trainer.data_processor
            self.is_trained = True
            
            # Save the trained model
            self.trainer.save_model()
            
            # Compute hydrocarbon statistics
            self._compute_hydrocarbon_stats(df)
            
            print("Model training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
    
    def _load_pretrained_model(self):
        """Load a pre-trained model"""
        try:
            if self.trainer.load_model():
                self.model = self.trainer.best_model
                self.data_processor = self.trainer.data_processor
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"Error loading pre-trained model: {str(e)}")
            return False
    
    def _compute_hydrocarbon_stats(self, df):
        """Compute statistics for each hydrocarbon"""
        for hydrocarbon in df['Hydrocarbon'].unique():
            stats = self.data_processor.get_hydrocarbon_stats(df, hydrocarbon)
            if stats:
                self.hydrocarbon_stats[hydrocarbon] = stats
    
    def predict(self, hydrocarbon, temperature, equivalence_ratio, pressure):
        """
        Make a single LBV prediction
        
        Args:
            hydrocarbon: Name of the hydrocarbon
            temperature: Initial temperature in Kelvin
            equivalence_ratio: Equivalence ratio (phi)
            pressure: Pressure in atmospheres
            
        Returns:
            Predicted LBV in cm/s or None if prediction fails
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            # Validate inputs
            if not self._validate_inputs(hydrocarbon, temperature, equivalence_ratio, pressure):
                return None
            
            # Make prediction using trainer
            prediction = self.trainer.predict_single(
                hydrocarbon, temperature, equivalence_ratio, pressure
            )
            
            # Ensure prediction is positive
            prediction = max(0, prediction)
            
            return float(prediction)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def _validate_inputs(self, hydrocarbon, temperature, equivalence_ratio, pressure):
        """Validate input parameters"""
        try:
            # Check if hydrocarbon is known
            if hydrocarbon not in self.hydrocarbon_stats:
                print(f"Unknown hydrocarbon: {hydrocarbon}")
                return False
            
            stats = self.hydrocarbon_stats[hydrocarbon]
            
            # Check temperature range
            temp_min, temp_max = stats['temperature_range']
            if not (temp_min <= temperature <= temp_max):
                print(f"Temperature {temperature} K is outside valid range [{temp_min}, {temp_max}] for {hydrocarbon}")
                return False
            
            # Check equivalence ratio range
            ratio_min, ratio_max = stats['ratio_range']
            if not (ratio_min <= equivalence_ratio <= ratio_max):
                print(f"Equivalence ratio {equivalence_ratio} is outside valid range [{ratio_min:.2f}, {ratio_max:.2f}] for {hydrocarbon}")
                return False
            
            # Check pressure range
            pressure_min, pressure_max = stats['pressure_range']
            if not (pressure_min <= pressure <= pressure_max):
                print(f"Pressure {pressure} atm is outside valid range [{pressure_min}, {pressure_max}] for {hydrocarbon}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False
    
    def get_hydrocarbon_ranges(self, hydrocarbon):
        """Get valid parameter ranges for a specific hydrocarbon"""
        if hydrocarbon in self.hydrocarbon_stats:
            return self.hydrocarbon_stats[hydrocarbon]
        return None
    
    def get_available_hydrocarbons(self):
        """Get list of available hydrocarbons"""
        return list(self.hydrocarbon_stats.keys())
    
    def predict_batch(self, input_df):
        """
        Make batch predictions
        
        Args:
            input_df: DataFrame with columns ['Hydrocarbon', 'Ti (K)', 'equivalent ratio', 'Pressure (atm)']
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            # Prepare features without fitting (use existing encoders/scalers)
            X, _ = self.data_processor.prepare_features(input_df, fit=False)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Ensure all predictions are positive
            predictions = np.maximum(0, predictions)
            
            return predictions
            
        except Exception as e:
            print(f"Batch prediction error: {str(e)}")
            return None
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return None
        
        info = {
            'model_type': self.trainer.best_model_name if hasattr(self.trainer, 'best_model_name') else 'Unknown',
            'training_scores': getattr(self.trainer, 'training_scores', {}),
            'available_hydrocarbons': len(self.hydrocarbon_stats),
            'hydrocarbon_list': list(self.hydrocarbon_stats.keys())
        }
        
        return info
