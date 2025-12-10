"""
Setup Verification Script
Checks if the Comment Categorization Tool is properly set up
"""

import sys
import os


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_python_version():
    """Check if Python version is adequate"""
    print("\n📌 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version is adequate (3.8+)")
        return True
    else:
        print("   ❌ Python 3.8 or higher is required")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print("\n📌 Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'nltk': 'nltk',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'wordcloud': 'wordcloud'
    }
    
    missing = []
    installed = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            installed.append(pip_name)
            print(f"   ✅ {pip_name}")
        except ImportError:
            missing.append(pip_name)
            print(f"   ❌ {pip_name} - NOT INSTALLED")
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    else:
        print(f"\n   ✅ All {len(installed)} required packages are installed!")
        return True


def check_nltk_data():
    """Check if NLTK data is downloaded"""
    print("\n📌 Checking NLTK data...")
    
    try:
        import nltk
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        missing = []
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else 
                              f'taggers/{data}' if 'tagger' in data else 
                              f'corpora/{data}')
                print(f"   ✅ {data}")
            except LookupError:
                missing.append(data)
                print(f"   ❌ {data} - NOT DOWNLOADED")
        
        if missing:
            print(f"\n   Missing NLTK data: {', '.join(missing)}")
            print("   Download with: python -c \"import nltk; nltk.download('all')\"")
            return False
        else:
            print(f"\n   ✅ All NLTK data is available!")
            return True
            
    except ImportError:
        print("   ❌ NLTK not installed")
        return False


def check_files():
    """Check if all required files exist"""
    print("\n📌 Checking project files...")
    
    required_files = [
        'dataset.csv',
        'preprocessor.py',
        'model_trainer.py',
        'response_templates.py',
        'app.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing = []
    base_path = 'd:/ASSIGNMENT/'
    
    for file in required_files:
        filepath = os.path.join(base_path, file)
        if os.path.exists(filepath):
            print(f"   ✅ {file}")
        else:
            missing.append(file)
            print(f"   ❌ {file} - NOT FOUND")
    
    if missing:
        print(f"\n   Missing files: {', '.join(missing)}")
        return False
    else:
        print(f"\n   ✅ All {len(required_files)} required files exist!")
        return True


def check_model():
    """Check if the model is trained"""
    print("\n📌 Checking trained model...")
    
    model_path = 'd:/ASSIGNMENT/comment_classifier.pkl'
    
    if os.path.exists(model_path):
        print(f"   ✅ Model file exists: comment_classifier.pkl")
        
        # Try loading the model
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"   ✅ Model can be loaded successfully")
            print(f"   ✅ Model type: {model_data.get('model_type', 'Unknown')}")
            return True
        except Exception as e:
            print(f"   ⚠️  Model file exists but cannot be loaded: {e}")
            print(f"   💡 Try retraining: python model_trainer.py")
            return False
    else:
        print(f"   ❌ Model not found")
        print(f"   💡 Train the model first: python model_trainer.py")
        return False


def check_dataset():
    """Check dataset quality"""
    print("\n📌 Checking dataset...")
    
    try:
        import pandas as pd
        df = pd.read_csv('d:/ASSIGNMENT/dataset.csv')
        
        print(f"   ✅ Dataset loaded successfully")
        print(f"   ✅ Total comments: {len(df)}")
        
        if 'comment' in df.columns and 'category' in df.columns:
            print(f"   ✅ Required columns present")
        else:
            print(f"   ❌ Missing required columns")
            return False
        
        categories = df['category'].unique()
        print(f"   ✅ Categories: {len(categories)}")
        
        required_categories = [
            'Praise', 'Support', 'Constructive Criticism',
            'Hate', 'Threat', 'Emotional', 'Spam', 'Question'
        ]
        
        missing_cats = []
        for cat in required_categories:
            if cat in categories:
                count = len(df[df['category'] == cat])
                print(f"      • {cat}: {count} examples")
            else:
                missing_cats.append(cat)
                print(f"      ❌ {cat}: MISSING")
        
        if missing_cats:
            print(f"\n   ⚠️  Missing categories: {', '.join(missing_cats)}")
            return False
        else:
            print(f"\n   ✅ All required categories present!")
            return True
            
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False


def test_imports():
    """Test if modules can be imported"""
    print("\n📌 Testing module imports...")
    
    modules = [
        ('preprocessor', 'TextPreprocessor'),
        ('model_trainer', 'CommentClassifier'),
        ('response_templates', 'ResponseTemplates')
    ]
    
    success = True
    
    for module_name, class_name in modules:
        try:
            sys.path.insert(0, 'd:/ASSIGNMENT')
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name} - {e}")
            success = False
    
    if success:
        print(f"\n   ✅ All modules can be imported!")
    
    return success


def run_quick_test():
    """Run a quick classification test"""
    print("\n📌 Running quick classification test...")
    
    try:
        sys.path.insert(0, 'd:/ASSIGNMENT')
        from model_trainer import CommentClassifier
        
        # Check if model exists
        if not os.path.exists('d:/ASSIGNMENT/comment_classifier.pkl'):
            print("   ⚠️  Model not trained yet - skipping test")
            return True
        
        # Load model
        classifier = CommentClassifier.load_model('d:/ASSIGNMENT/comment_classifier.pkl')
        print("   ✅ Model loaded")
        
        # Test prediction
        test_comment = "Amazing work! Loved it!"
        prediction = classifier.predict([test_comment])[0]
        confidence = classifier.predict_proba([test_comment])[0].max()
        
        print(f"   ✅ Test prediction successful")
        print(f"      Comment: '{test_comment}'")
        print(f"      Predicted: {prediction}")
        print(f"      Confidence: {confidence:.2%}")
        
        if prediction == "Praise":
            print("   ✅ Correct prediction!")
        else:
            print(f"   ⚠️  Expected 'Praise', got '{prediction}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def main():
    """Main verification function"""
    
    print_section("Comment Categorization Tool - Setup Verification")
    print("\nThis script will verify that everything is set up correctly.")
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("NLTK Data", check_nltk_data()))
    results.append(("Project Files", check_files()))
    results.append(("Dataset", check_dataset()))
    results.append(("Module Imports", test_imports()))
    results.append(("Trained Model", check_model()))
    results.append(("Quick Test", run_quick_test()))
    
    # Summary
    print_section("Verification Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n   Tests Passed: {passed}/{total}")
    print()
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print("\n" + "="*70)
    
    if passed == total:
        print("\n🎉 SUCCESS! Everything is set up correctly!")
        print("\n📋 Next Steps:")
        print("   1. Run: python model_trainer.py (if model not trained)")
        print("   2. Run: streamlit run app.py")
        print("   3. Or try: python example_usage.py")
    else:
        print("\n⚠️  ISSUES FOUND! Please fix the failed checks above.")
        print("\n💡 Quick Fixes:")
        print("   • Install dependencies: pip install -r requirements.txt")
        print("   • Download NLTK data: python -c \"import nltk; nltk.download('all')\"")
        print("   • Train model: python model_trainer.py")
    
    print("\n" + "="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
