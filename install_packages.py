"""
Install Package Script
Quickly install all dependencies for the Comment Categorization Tool
"""

import subprocess
import sys


def install_packages():
    """Install all required packages"""
    
    print("="*70)
    print("Installing Dependencies for Comment Categorization Tool")
    print("="*70)
    print()
    
    try:
        print("📦 Installing packages from requirements.txt...")
        print()
        
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "requirements.txt",
            "--upgrade"
        ])
        
        print()
        print("="*70)
        print("✅ All packages installed successfully!")
        print("="*70)
        print()
        
        print("📚 Downloading NLTK data...")
        
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        print("✅ NLTK data downloaded!")
        print()
        
        print("="*70)
        print("🎉 Setup Complete!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Train the model: python model_trainer.py")
        print("  2. Launch the app:  streamlit run app.py")
        print("  3. Or run example:   python example_usage.py")
        print()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print("="*70)
        print("❌ Installation failed!")
        print("="*70)
        print(f"Error: {e}")
        print()
        print("Please try manually:")
        print("  pip install -r requirements.txt")
        print()
        return False
    
    except Exception as e:
        print()
        print("="*70)
        print("❌ Unexpected error!")
        print("="*70)
        print(f"Error: {e}")
        print()
        return False


if __name__ == "__main__":
    success = install_packages()
    sys.exit(0 if success else 1)
