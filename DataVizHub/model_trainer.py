import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import joblib
from data_processor import DataProcessor

class ModelTrainer:
    """Train and evaluate LBV prediction models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.data_processor = DataProcessor()
        self.training_scores = {}
    
    def train_models(self, df, test_size=0.2, random_state=42):
        """
        Train multiple models and select the best one
        
        Args:
            df: Preprocessed DataFrame
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Prepare features
        X, y = self.data_processor.prepare_features(df, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Define models to train
        model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        best_score = -np.inf
        
        for model_name, config in model_configs.items():
            print(f"\nTraining {model_name}...")
            
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            self.models[model_name] = best_model
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            self.training_scores[model_name] = {
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'best_params': grid_search.best_params_
            }
            
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            print(f"Best params: {grid_search.best_params_}")
            
            # Update best model
            if r2 > best_score:
                best_score = r2
                self.best_model = best_model
                self.best_model_name = model_name
        
        print(f"\nBest model: {self.best_model_name} (R² = {best_score:.4f})")
        
        # Feature importance analysis
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = ['Temperature', 'Equivalence Ratio', 'Pressure', 'Hydrocarbon']
            importances = self.best_model.feature_importances_
            
            print("\nFeature Importances:")
            for name, importance in zip(feature_names, importances):
                print(f"{name}: {importance:.4f}")
        
        return self.best_model, self.training_scores
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a trained model"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics, y_pred
    
    def save_model(self, model_path='lbv_model.pkl', processor_path='preprocessors.pkl'):
        """Save the best trained model and preprocessors"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Save model
        joblib.dump(self.best_model, model_path)
        
        # Save preprocessors
        self.data_processor.save_preprocessors(processor_path)
        
        # Save training metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'training_scores': self.training_scores
        }
        
        with open('training_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {model_path}")
        print(f"Preprocessors saved to {processor_path}")
    
    def load_model(self, model_path='lbv_model.pkl', processor_path='preprocessors.pkl'):
        """Load a trained model and preprocessors"""
        try:
            # Load model
            self.best_model = joblib.load(model_path)
            
            # Load preprocessors
            self.data_processor.load_preprocessors(processor_path)
            
            # Load metadata if available
            try:
                with open('training_metadata.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                self.best_model_name = metadata.get('best_model_name', 'unknown')
                self.training_scores = metadata.get('training_scores', {})
            except FileNotFoundError:
                print("Training metadata not found")
            
            print(f"Model loaded from {model_path}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {model_path} not found")
            return False
    
    def predict_single(self, hydrocarbon, temperature, equivalence_ratio, pressure):
        """Make a single prediction"""
        if self.best_model is None:
            raise ValueError("No model loaded")
        
        # Encode hydrocarbon
        hydrocarbon_encoded = self.data_processor.encode_hydrocarbon(hydrocarbon)
        
        # Prepare features
        features = np.array([[temperature, equivalence_ratio, pressure, hydrocarbon_encoded]])
        
        # Scale features
        features_scaled = self.data_processor.scaler.transform(features)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        
        return prediction
