@echo off
echo ============================================================
echo Comment Categorization Tool - Automatic Setup
echo ============================================================
echo.

echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo ✓ Python is installed
echo.

echo Step 2: Installing required packages...
echo This may take a few minutes...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo ✓ Packages installed successfully
echo.

echo Step 3: Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
if errorlevel 1 (
    echo WARNING: NLTK download had issues, but continuing...
)
echo ✓ NLTK data downloaded
echo.

echo Step 4: Training the classification model...
echo This will take 1-2 minutes...
echo.
python model_trainer.py
if errorlevel 1 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)
echo ✓ Model trained successfully
echo.

echo Step 5: Running tests...
python test_system.py
echo.

echo ============================================================
echo Setup Complete! 🎉
echo ============================================================
echo.
echo The Comment Categorization Tool is ready to use!
echo.
echo To start the web application, run:
echo     streamlit run app.py
echo.
echo Or try the example script:
echo     python example_usage.py
echo.
echo For more information, see:
echo   - README.md (full documentation)
echo   - QUICKSTART.md (quick start guide)
echo.
echo ============================================================
pause
