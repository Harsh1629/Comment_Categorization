"""
Model Training Module for Comment Classification
Implements multiple ML models and evaluation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessor import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')


class CommentClassifier:
    """
    Multi-model comment classification system
    """
    
    def __init__(self, model_type='logistic_regression', use_tfidf=True):
        """
        Initialize the classifier
        
        Args:
            model_type (str): Type of model to use
            use_tfidf (bool): Use TF-IDF vs Count Vectorizer
        """
        self.model_type = model_type
        self.use_tfidf = use_tfidf
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.pipeline = None
        self.categories = None
        
    def create_model(self):
        """
        Create the classification model based on type
        
        Returns:
            sklearn model: Initialized model
        """
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='linear',
                C=1.0,
                random_state=42,
                probability=True,
                class_weight='balanced'
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'ensemble':
            lr = LogisticRegression(max_iter=1000, random_state=42)
            svm = SVC(kernel='linear', probability=True, random_state=42)
            nb = MultinomialNB()
            return VotingClassifier(
                estimators=[('lr', lr), ('svm', svm), ('nb', nb)],
                voting='soft'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def create_vectorizer(self):
        """
        Create text vectorizer
        
        Returns:
            sklearn vectorizer: TF-IDF or Count Vectorizer
        """
        if self.use_tfidf:
            return TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
        else:
            return CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
    
    def train(self, X_train, y_train):
        """
        Train the classification model
        
        Args:
            X_train (list): Training texts
            y_train (list): Training labels
        """
        # Preprocess training data
        X_train_processed = [self.preprocessor.preprocess(text) for text in X_train]
        
        # Create pipeline
        self.vectorizer = self.create_vectorizer()
        self.model = self.create_model()
        
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # Train the model
        print(f"Training {self.model_type} model...")
        self.pipeline.fit(X_train_processed, y_train)
        print("Training completed!")
        
        # Store categories
        self.categories = list(self.pipeline.classes_)
    
    def predict(self, texts):
        """
        Predict categories for new texts
        
        Args:
            texts (list): List of texts to classify
            
        Returns:
            list: Predicted categories
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        # Preprocess texts
        texts_processed = [self.preprocessor.preprocess(text) for text in texts]
        
        # Predict
        predictions = self.pipeline.predict(texts_processed)
        return predictions
    
    def predict_proba(self, texts):
        """
        Predict probabilities for each category
        
        Args:
            texts (list): List of texts to classify
            
        Returns:
            np.array: Probability matrix
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        # Preprocess texts
        texts_processed = [self.preprocessor.preprocess(text) for text in texts]
        
        # Predict probabilities
        probabilities = self.pipeline.predict_proba(texts_processed)
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (list): Test texts
            y_test (list): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Preprocess test data
        X_test_processed = [self.preprocessor.preprocess(text) for text in X_test]
        
        # Predictions
        y_pred = self.pipeline.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"\n{'='*50}")
        print(f"Model: {self.model_type}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plot confusion matrix
        
        Args:
            y_test (list): True labels
            y_pred (list): Predicted labels
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.categories,
                    yticklabels=self.categories)
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'd:/ASSIGNMENT/confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved as confusion_matrix_{self.model_type}.png")
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'pipeline': self.pipeline,
            'model_type': self.model_type,
            'categories': self.categories,
            'preprocessor': self.preprocessor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load trained model from disk
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            CommentClassifier: Loaded classifier
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = CommentClassifier(model_type=model_data['model_type'])
        classifier.pipeline = model_data['pipeline']
        classifier.categories = model_data['categories']
        classifier.preprocessor = model_data['preprocessor']
        
        return classifier


def train_multiple_models(X_train, X_test, y_train, y_test):
    """
    Train and compare multiple models
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        
    Returns:
        dict: Results for all models
    """
    models = ['logistic_regression', 'svm', 'naive_bayes', 'random_forest']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type}...")
        print(f"{'='*60}")
        
        classifier = CommentClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        classifier.plot_confusion_matrix(y_test, metrics['predictions'])
        
        results[model_type] = {
            'classifier': classifier,
            'metrics': metrics
        }
    
    return results


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('d:/ASSIGNMENT/dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    
    # Split data
    X = df['comment'].values
    y = df['category'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Train single best model (Logistic Regression)
    print("\n" + "="*60)
    print("Training Best Model (Logistic Regression)...")
    print("="*60)
    
    classifier = CommentClassifier(model_type='logistic_regression')
    classifier.train(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)
    classifier.plot_confusion_matrix(y_test, metrics['predictions'])
    classifier.save_model('d:/ASSIGNMENT/comment_classifier.pkl')
    
    print("\n" + "="*60)
    print("Model training completed successfully!")
    print("="*60)
