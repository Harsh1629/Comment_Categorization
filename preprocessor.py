"""
Text Preprocessing Module for Comment Classification
Handles cleaning, normalization, and feature extraction
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for comment analysis
    """
    
    def __init__(self, lowercase=True, remove_stopwords=True, lemmatize=True):
        """
        Initialize the preprocessor with configuration options
        
        Args:
            lowercase (bool): Convert text to lowercase
            remove_stopwords (bool): Remove common stopwords
            lemmatize (bool): Apply lemmatization
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean text by removing URLs, mentions, hashtags, and special characters
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_punctuation(self, text):
        """
        Remove punctuation while preserving sentence structure
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without punctuation
        """
        # Keep some punctuation that might be meaningful (!, ?)
        translator = str.maketrans('', '', string.punctuation.replace('!', '').replace('?', ''))
        return text.translate(translator)
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_func(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [word for word in tokens if word.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        text = self.remove_punctuation(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='comment'):
        """
        Preprocess all texts in a dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        df = df.copy()
        df['processed_text'] = df[text_column].apply(self.preprocess)
        return df


def extract_features(text):
    """
    Extract additional features from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of features
    """
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'has_url': 1 if re.search(r'http\S+|www\S+', text) else 0,
        'has_mention': 1 if re.search(r'@\w+', text) else 0,
        'has_hashtag': 1 if re.search(r'#\w+', text) else 0
    }
    return features


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "Amazing work! Loved the animation.",
        "This is trash, quit now.",
        "The animation was okay but the voiceover felt off.",
        "Follow me for followers! Check my link!"
    ]
    
    print("Testing Preprocessor:\n")
    for text in test_texts:
        processed = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print(f"Features: {extract_features(text)}")
        print("-" * 80)
