"""
Fraud Detection Model Training Script
Trains Random Forest model and saves it as pickle for deployment
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_and_save_model(data_path='Fraud.csv'):
    """
    Train the fraud detection model and save artifacts
    """
    print("=" * 60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Data preprocessing
    print("\n[2/6] Preprocessing data...")
    
    # Handle missing values
    df['oldbalanceDest'].fillna(0, inplace=True)
    df['newbalanceDest'].fillna(0, inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Feature engineering
    df['diffOrig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['diffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['isLargeTxn'] = (df['amount'] > 200000).astype(int)
    
    # One-hot encode transaction type
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    print("✓ Feature engineering completed")
    
    # Prepare features and target
    print("\n[3/6] Preparing features...")
    X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
    y = df['isFraud']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train set: {len(X_train):,} | Test set: {len(X_test):,}")
    
    # Scale numeric features
    print("\n[4/6] Scaling features...")
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest', 'diffOrig', 'diffDest']
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    print("✓ Features scaled")
    
    # Train model
    print("\n[5/6] Training Random Forest model...")
    print("This may take several minutes...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    print("✓ Model training completed")
    
    # Evaluate
    print("\n[6/6] Evaluating model performance...")
    y_pred = rf_model.predict(X_test)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_score = roc_auc_score(y_test, y_pred)
    print(f"ROC-AUC Score: {roc_score:.4f}")
    
    # Save model artifacts
    print("\n" + "=" * 60)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/fraud_detector.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = 'models/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = 'models/feature_names.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved: {features_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'features': feature_names,
        'numeric_cols': numeric_cols,
        'roc_auc_score': roc_score,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    metadata_path = 'models/metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE - All artifacts saved successfully!")
    print("=" * 60)
    print("\nSaved files:")
    print("  - models/fraud_detector.pkl")
    print("  - models/scaler.pkl")
    print("  - models/feature_names.pkl")
    print("  - models/metadata.pkl")
    print("\nYou can now run the Django app!")
    
    return rf_model, scaler, feature_names

if __name__ == '__main__':
    train_and_save_model()
