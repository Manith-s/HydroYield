"""
Preprocess data for machine learning model training
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_data(input_path, output_path=None):
    """
    Preprocess the crop dataset for model training
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the processed data
        
    Returns:
        tuple: Processed X (features), y (labels)
    """
    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Display basic info
        logger.info(f"Original dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df.dtypes}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        logger.info(f"Missing values:\n{missing_values}")
        
        if missing_values.sum() > 0:
            # Handle missing values
            logger.info("Handling missing values...")
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        logger.info(f"Duplicate rows: {duplicates}")
        
        if duplicates > 0:
            # Remove duplicates
            df = df.drop_duplicates()
            logger.info(f"Dataset shape after removing duplicates: {df.shape}")
        
        # Handle outliers using IQR method
        logger.info("Handling outliers...")
        numerical_cols = ['temperature', 'humidity', 'ph', 'rainfall']
        
        for col in numerical_cols:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            logger.info(f"Outliers in {col}: {outliers}")
            
            # Cap outliers instead of removing them
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        # Encode target variable
        logger.info("Encoding target variable...")
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df['label'])
        
        # Save mapping for future reference
        label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
        logger.info(f"Label mapping: {label_mapping}")
        
        # Save mapping to file
        if output_path:
            mapping_path = os.path.join(os.path.dirname(output_path), 'label_mapping.csv')
            pd.DataFrame({
                'index': list(label_mapping.keys()),
                'crop': list(label_mapping.values())
            }).to_csv(mapping_path, index=False)
            logger.info(f"Label mapping saved to {mapping_path}")
        
        # Feature scaling
        logger.info("Performing feature scaling...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[numerical_cols])
        
        # Create a new DataFrame with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols)
        
        # Add the target variable back
        scaled_df['label'] = df['label']
        scaled_df['encoded_label'] = df['encoded_label']
        
        # Display statistics of processed data
        logger.info(f"Processed dataset shape: {scaled_df.shape}")
        logger.info(f"Processed data statistics:\n{scaled_df.describe()}")
        
        # Save processed data if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            scaled_df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            
            # Also save the scaler
            scaler_path = os.path.join(os.path.dirname(output_path), 'scaler.pkl')
            import joblib
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Return features and labels
        X = scaled_df[numerical_cols]
        y = scaled_df['label']
        
        return X, y, label_mapping
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

if __name__ == "__main__":
    try:
        # Parse command-line arguments
        if len(sys.argv) >= 3:
            input_path = sys.argv[1]
            output_path = sys.argv[2]
        elif len(sys.argv) == 2:
            input_path = sys.argv[1]
            output_path = os.path.join("data", "processed_data.csv")
        else:
            # Default paths
            input_path = os.path.join("data", "cpdata.csv")
            output_path = os.path.join("data", "processed_data.csv")
        
        # Preprocess data
        X, y, label_mapping = preprocess_data(input_path, output_path)
        
        logger.info("Preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)