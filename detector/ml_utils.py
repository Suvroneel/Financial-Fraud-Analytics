"""
ML Model Utility
Handles loading and using the fraud detection model
"""

import pickle
import pandas as pd
import numpy as np
from django.conf import settings
import os


class FraudDetector:
    """
    Singleton class to load and cache the ML model
    """
    _instance = None
    _model = None
    _scaler = None
    _feature_names = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FraudDetector, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load model artifacts from disk"""
        try:
            # Load model
            with open(settings.MODEL_PATH, 'rb') as f:
                self._model = pickle.load(f)
            
            # Load scaler
            with open(settings.SCALER_PATH, 'rb') as f:
                self._scaler = pickle.load(f)
            
            # Load feature names
            with open(settings.FEATURE_NAMES_PATH, 'rb') as f:
                self._feature_names = pickle.load(f)
            
            # Load metadata
            with open(settings.METADATA_PATH, 'rb') as f:
                self._metadata = pickle.load(f)
            
            print("✓ Model loaded successfully!")
            print(f"  Features: {len(self._feature_names)}")
            print(f"  Model: {self._metadata.get('model_type', 'Unknown')}")
            
        except FileNotFoundError as e:
            print(f"ERROR: Model files not found!")
            print(f"Please run 'python train_model.py' first to train and save the model.")
            raise
        except Exception as e:
            print(f"ERROR loading model: {str(e)}")
            raise
    
    def predict(self, transaction_data):
        """
        Make fraud prediction on a single transaction
        
        Args:
            transaction_data: dict with keys:
                - amount: float
                - oldbalanceOrg: float
                - newbalanceOrig: float
                - oldbalanceDest: float
                - newbalanceDest: float
                - type: str (CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT)
        
        Returns:
            dict with prediction result and probability
        """
        try:
            # Feature engineering (same as training)
            data = transaction_data.copy()
            data['diffOrig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
            data['diffDest'] = data['newbalanceDest'] - data['oldbalanceDest']
            data['isLargeTxn'] = 1 if data['amount'] > 200000 else 0
            
            # One-hot encode transaction type
            transaction_type = data.pop('type')
            type_columns = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
            for col in type_columns:
                data[col] = 0
            
            if transaction_type == 'CASH_OUT':
                data['type_CASH_OUT'] = 1
            elif transaction_type == 'DEBIT':
                data['type_DEBIT'] = 1
            elif transaction_type == 'PAYMENT':
                data['type_PAYMENT'] = 1
            elif transaction_type == 'TRANSFER':
                data['type_TRANSFER'] = 1
            
            # Create dataframe with correct column order
            df = pd.DataFrame([data])
            df = df[self._feature_names]
            
            # Scale numeric features
            numeric_cols = self._metadata['numeric_cols']
            df[numeric_cols] = self._scaler.transform(df[numeric_cols])
            
            # Make prediction
            prediction = self._model.predict(df)[0]
            probability = self._model.predict_proba(df)[0]
            
            return {
                'is_fraud': bool(prediction),
                'fraud_probability': float(probability[1]),
                'confidence': float(max(probability))
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
    
    def get_model_info(self):
        """Return model metadata"""
        return self._metadata
