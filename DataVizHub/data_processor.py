import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

class DataProcessor:
    """Data processing class for LBV prediction dataset"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset for machine learning
        
        Args:
            df: Raw pandas DataFrame
            
        Returns:
            Processed pandas DataFrame
        """
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Clean column names
        processed_df.columns = processed_df.columns.str.strip()
        
        # Handle missing values
        processed_df = processed_df.dropna()
        
        # Data type conversions
        numeric_columns = ['Ti (K)', 'equivalent ratio', 'Pressure (atm)', 'LBV (cm/s)']
        
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Remove any rows with invalid numeric values
        processed_df = processed_df.dropna()
        
        # Remove outliers using IQR method for LBV values
        Q1 = processed_df['LBV (cm/s)'].quantile(0.25)
        Q3 = processed_df['LBV (cm/s)'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        processed_df = processed_df[
            (processed_df['LBV (cm/s)'] >= lower_bound) & 
            (processed_df['LBV (cm/s)'] <= upper_bound)
        ]
        
        # Validate data ranges
        processed_df = self._validate_ranges(processed_df)
        
        return processed_df
    
    def _validate_ranges(self, df):
        """Validate that data is within reasonable physical ranges"""
        # Temperature should be reasonable (200K to 1000K)
        df = df[(df['Ti (K)'] >= 200) & (df['Ti (K)'] <= 1000)]
        
        # Equivalence ratio should be reasonable (0.1 to 3.0)
        df = df[(df['equivalent ratio'] >= 0.1) & (df['equivalent ratio'] <= 3.0)]
        
        # Pressure should be reasonable (0.1 to 50 atm)
        df = df[(df['Pressure (atm)'] >= 0.1) & (df['Pressure (atm)'] <= 50)]
        
        # LBV should be positive and reasonable (up to 500 cm/s)
        df = df[(df['LBV (cm/s)'] > 0) & (df['LBV (cm/s)'] <= 500)]
        
        return df
    
    def prepare_features(self, df, fit=True):
        """
        Prepare features for machine learning
        
        Args:
            df: Preprocessed DataFrame
            fit: Whether to fit the encoders/scalers
            
        Returns:
            X: Feature matrix
            y: Target vector (if present)
        """
        # Features
        feature_columns = ['Hydrocarbon', 'Ti (K)', 'equivalent ratio', 'Pressure (atm)']
        X = df[feature_columns].copy()
        
        # Encode categorical variables
        if fit:
            X['Hydrocarbon_encoded'] = self.label_encoder.fit_transform(X['Hydrocarbon'])
        else:
            # Handle unseen labels
            X['Hydrocarbon_encoded'] = X['Hydrocarbon'].apply(
                lambda x: self.label_encoder.transform([x])[0] 
                if x in self.label_encoder.classes_ 
                else -1
            )
        
        # Prepare numerical features
        numerical_features = ['Ti (K)', 'equivalent ratio', 'Pressure (atm)', 'Hydrocarbon_encoded']
        X_numerical = X[numerical_features]
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_numerical)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler has not been fitted yet")
            X_scaled = self.scaler.transform(X_numerical)
        
        # Target variable (if present)
        y = None
        if 'LBV (cm/s)' in df.columns:
            y = df['LBV (cm/s)'].values
        
        return X_scaled, y
    
    def encode_hydrocarbon(self, hydrocarbon_name):
        """Encode a single hydrocarbon name"""
        if hydrocarbon_name in self.label_encoder.classes_:
            return self.label_encoder.transform([hydrocarbon_name])[0]
        else:
            return -1  # Unknown hydrocarbon
    
    def get_hydrocarbon_stats(self, df, hydrocarbon_name):
        """Get statistics for a specific hydrocarbon"""
        hydrocarbon_data = df[df['Hydrocarbon'] == hydrocarbon_name]
        
        if hydrocarbon_data.empty:
            return None
        
        stats = {
            'temperature_range': (
                float(hydrocarbon_data['Ti (K)'].min()), 
                float(hydrocarbon_data['Ti (K)'].max())
            ),
            'ratio_range': (
                float(hydrocarbon_data['equivalent ratio'].min()), 
                float(hydrocarbon_data['equivalent ratio'].max())
            ),
            'pressure_range': (
                float(hydrocarbon_data['Pressure (atm)'].min()), 
                float(hydrocarbon_data['Pressure (atm)'].max())
            ),
            'lbv_range': (
                float(hydrocarbon_data['LBV (cm/s)'].min()), 
                float(hydrocarbon_data['LBV (cm/s)'].max())
            ),
            'data_points': len(hydrocarbon_data)
        }
        
        return stats
    
    def save_preprocessors(self, filepath='preprocessors.pkl'):
        """Save the fitted preprocessors"""
        preprocessors = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
    
    def load_preprocessors(self, filepath='preprocessors.pkl'):
        """Load the fitted preprocessors"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                preprocessors = pickle.load(f)
            
            self.label_encoder = preprocessors['label_encoder']
            self.scaler = preprocessors['scaler']
            self.is_fitted = preprocessors['is_fitted']
            
            return True
        return False
