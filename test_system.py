"""
Unit Tests for Comment Classification System
"""

import unittest
import pandas as pd
from preprocessor import TextPreprocessor, extract_features
from model_trainer import CommentClassifier
from response_templates import ResponseTemplates


class TestPreprocessor(unittest.TestCase):
    """Test cases for the preprocessor module"""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        text = "Check out http://example.com @user #hashtag test@email.com"
        cleaned = self.preprocessor.clean_text(text)
        
        self.assertNotIn('http://', cleaned)
        self.assertNotIn('@user', cleaned)
        self.assertNotIn('#hashtag', cleaned)
        self.assertNotIn('test@email.com', cleaned)
    
    def test_preprocess(self):
        """Test complete preprocessing pipeline"""
        text = "This is AMAZING! I really loved it!!!"
        processed = self.preprocessor.preprocess(text)
        
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)
        # Should be lowercase
        self.assertEqual(processed, processed.lower())
    
    def test_extract_features(self):
        """Test feature extraction"""
        text = "This is great! What do you think?"
        features = extract_features(text)
        
        self.assertIn('length', features)
        self.assertIn('word_count', features)
        self.assertIn('exclamation_count', features)
        self.assertIn('question_count', features)
        
        self.assertEqual(features['exclamation_count'], 1)
        self.assertEqual(features['question_count'], 1)


class TestResponseTemplates(unittest.TestCase):
    """Test cases for response templates"""
    
    def setUp(self):
        self.rt = ResponseTemplates()
    
    def test_get_templates(self):
        """Test getting all templates"""
        templates = self.rt.get_templates()
        
        self.assertIsInstance(templates, dict)
        self.assertIn('Praise', templates)
        self.assertIn('Constructive Criticism', templates)
        self.assertIn('Threat', templates)
    
    def test_get_response(self):
        """Test getting specific response"""
        response = self.rt.get_response('Praise', 0)
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_get_action_guide(self):
        """Test getting action guide"""
        guide = self.rt.get_action_guide()
        
        self.assertIsInstance(guide, dict)
        self.assertIn('priority', guide['Praise'])
        self.assertIn('action', guide['Praise'])
        self.assertIn('notes', guide['Praise'])
    
    def test_priority_order(self):
        """Test priority order"""
        order = self.rt.get_priority_order()
        
        self.assertIsInstance(order, list)
        self.assertEqual(order[0], 'Threat')  # Threat should be highest priority


class TestModelClassifier(unittest.TestCase):
    """Test cases for the classifier (requires trained model)"""
    
    def setUp(self):
        """Load dataset for testing"""
        self.df = pd.read_csv('d:/ASSIGNMENT/dataset.csv')
        self.assertTrue(len(self.df) > 0, "Dataset should not be empty")
    
    def test_dataset_structure(self):
        """Test dataset has correct structure"""
        self.assertIn('comment', self.df.columns)
        self.assertIn('category', self.df.columns)
    
    def test_category_distribution(self):
        """Test all required categories are present"""
        required_categories = [
            'Praise', 'Support', 'Constructive Criticism',
            'Hate', 'Threat', 'Emotional', 'Spam', 'Question'
        ]
        
        actual_categories = self.df['category'].unique()
        
        for cat in required_categories:
            self.assertIn(cat, actual_categories, 
                         f"Category '{cat}' should be in dataset")
    
    def test_dataset_size(self):
        """Test dataset has sufficient examples"""
        self.assertGreaterEqual(len(self.df), 100, 
                               "Dataset should have at least 100 examples")


def run_tests():
    """Run all tests"""
    print("="*70)
    print("Running Unit Tests")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestResponseTemplates))
    suite.addTests(loader.loadTestsFromTestCase(TestModelClassifier))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
