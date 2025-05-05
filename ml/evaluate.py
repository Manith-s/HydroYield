"""
Evaluate the trained Random Forest model
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_data(model_path, data_path):
    """
    Load the trained model and test data
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the test data
        
    Returns:
        tuple: Model, features, labels
    """
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Split features and target
        feature_cols = ['temperature', 'humidity', 'ph', 'rainfall']
        X = data[feature_cols]
        y = data['label']
        
        # Ensure labels are encoded
        if y.dtype == 'object':
            logger.info("Encoding labels")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        return model, X, y
        
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        raise

def evaluate_model(model, X, y):
    """
    Evaluate the model performance
    
    Args:
        model: Trained model
        X: Features
        y: Target labels
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        # For multi-class, use 'weighted' average
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Get classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        # Store metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report
        }
        
        # Log results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"CV Mean: {cv_scores.mean():.4f}, CV Std: {cv_scores.std():.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def plot_confusion_matrix(model, X, y, class_names, output_dir):
    """
    Plot and save confusion matrix
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    try:
        # Make predictions
        y_pred = model.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")

def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot and save feature importance
    
    Args:
        model: Trained model
        feature_names: List of feature names
        output_dir: Directory to save the plot
    """
    try:
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame for plotting
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")

def plot_learning_curve(model, X, y, output_dir):
    """
    Plot and save learning curve
    
    Args:
        model: Trained model
        X: Features
        y: Target labels
        output_dir: Directory to save the plot
    """
    try:
        # Define train sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5, scoring='accuracy'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        
        # Plot standard deviation bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        # Labels and title
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curve plot saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting learning curve: {e}")

def save_metrics(metrics, output_dir):
    """
    Save evaluation metrics to file
    
    Args:
        metrics (dict): Evaluation metrics
        output_dir (str): Directory to save the metrics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON
        import json
        # Convert numpy values to Python native types
        metrics_json = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in metrics.items()}
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_json, f, indent=4)
        
        logger.info(f"Metrics saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")

if __name__ == "__main__":
    try:
        # Parse command-line arguments
        if len(sys.argv) >= 3:
            model_path = sys.argv[1]
            data_path = sys.argv[2]
            output_dir = sys.argv[3] if len(sys.argv) >= 4 else "ml/evaluation"
        else:
            # Default paths
            model_path = os.path.join("ml", "models", "random_forest_model.pkl")
            data_path = os.path.join("data", "cpdata.csv")
            output_dir = os.path.join("ml", "evaluation")
        
        # Load model and data
        model, X, y = load_model_and_data(model_path, data_path)
        
        # Get class names (assuming we have a mapping file)
        mapping_path = os.path.join(os.path.dirname(model_path), 'label_mapping.csv')
        if os.path.exists(mapping_path):
            mapping_df = pd.read_csv(mapping_path)
            class_names = mapping_df['crop'].tolist()
        else:
            # If no mapping file, use unique labels
            class_names = np.unique(y).tolist()
        
        # Evaluate model
        metrics = evaluate_model(model, X, y)
        
        # Plot visualizations
        plot_confusion_matrix(model, X, y, class_names, output_dir)
        plot_feature_importance(model, X.columns, output_dir)
        plot_learning_curve(model, X, y, output_dir)
        
        # Save metrics
        save_metrics(metrics, output_dir)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        sys.exit(1)