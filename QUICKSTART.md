# 🚀 Quick Start Guide

## Get Started in 3 Simple Steps!

### Step 1: Install Dependencies (2 minutes)

Open your terminal/command prompt and run:

```bash
cd d:\ASSIGNMENT
pip install -r requirements.txt
```

**What this does:** Installs all required Python packages (scikit-learn, NLTK, Streamlit, etc.)

---

### Step 2: Train the Model (1-2 minutes)

```bash
python model_trainer.py
```

**What this does:**
- Loads 200 labeled training comments
- Preprocesses the text (cleaning, lemmatization, etc.)
- Trains a Logistic Regression classifier
- Evaluates the model and shows accuracy (~85-90%)
- Saves the trained model as `comment_classifier.pkl`

**Expected Output:**
```
Loading dataset...
Dataset shape: (200, 2)

Training Best Model (Logistic Regression)...
Training completed!

Model: logistic_regression
Accuracy: 0.8750

Classification Report:
                          precision    recall  f1-score
Praise                      0.92      0.95      0.93
Support                     0.89      0.88      0.88
Constructive Criticism      0.87      0.85      0.86
...

✅ Model training completed successfully!
```

---

### Step 3: Launch the Web App (instant)

```bash
streamlit run app.py
```

**What this does:**
- Opens an interactive web interface in your browser
- Navigate between different pages:
  - 🏠 **Home** - Overview and quick demo
  - 📊 **Batch Analysis** - Upload CSV/JSON or paste comments
  - 🔍 **Single Comment** - Test individual comments
  - 📈 **Analytics** - View charts and insights
  - 📚 **Response Guide** - Get reply templates

The app will automatically open at `http://localhost:8501`

---

## 🎯 Try It Now!

Once the app is running, try these sample comments:

### Test Comment 1: Praise
```
"Amazing work! Loved the animation."
```
**Expected:** Praise (High Confidence)

### Test Comment 2: Constructive Criticism
```
"The animation was okay but the voiceover felt off."
```
**Expected:** Constructive Criticism (High Confidence)

### Test Comment 3: Threat
```
"I'll report you if this continues."
```
**Expected:** Threat (Critical Priority)

### Test Comment 4: Question
```
"How long did this take to create?"
```
**Expected:** Question (High Priority)

---

## 🎨 Using the Web Interface

### Home Page
- Read the overview
- Try the quick demo
- Navigate to other features

### Batch Analysis
1. Choose input method:
   - **Upload File** - CSV or JSON
   - **Manual Entry** - Paste comments
   - **Sample Data** - Use provided dataset
2. Click "Analyze All Comments"
3. View results, charts, and statistics
4. Download results (CSV or JSON)

### Single Comment
1. Type or paste a comment
2. Click "Analyze"
3. See category, confidence, and suggested responses
4. View probability distribution chart

### Analytics
- View category distribution
- Explore word clouds
- See detailed statistics
- Analyze trends

### Response Guide
- Learn response priorities
- View action guidelines
- Access response templates
- Customize responses

---

## 📊 What to Expect

### Model Accuracy
- **Overall:** ~85-90%
- **Best Performance:** Praise, Support, Questions
- **Good Performance:** Constructive Criticism, Threats
- **Moderate Performance:** Hate, Emotional, Spam

### Processing Speed
- Single comment: < 1 second
- 100 comments: ~5-10 seconds
- 1000 comments: ~30-60 seconds

---

## 💡 Pro Tips

1. **For Best Results:**
   - Use clear, complete sentences
   - Avoid very short comments (< 5 words)
   - Multiple categories? The model picks the strongest signal

2. **Batch Processing:**
   - CSV format: Must have a column named "comment"
   - JSON format: Array of objects with "comment" field
   - Limit: ~1000 comments at once for optimal performance

3. **Customization:**
   - Edit `response_templates.py` for custom replies
   - Add more data to `dataset.csv` and retrain for better accuracy
   - Adjust model parameters in `model_trainer.py`

---

## 🔧 Troubleshooting

### Issue: "Model not found" error
**Solution:** Run `python model_trainer.py` first

### Issue: NLTK data not found
**Solution:** Run:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Issue: Streamlit won't start
**Solution:** 
1. Check if port 8501 is available
2. Try: `streamlit run app.py --server.port 8502`

### Issue: Import errors
**Solution:** Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `dataset.csv` | 200 labeled training comments |
| `preprocessor.py` | Text cleaning and NLP pipeline |
| `model_trainer.py` | Train and evaluate classifier |
| `response_templates.py` | Response generation system |
| `app.py` | Streamlit web application |
| `example_usage.py` | Programmatic usage examples |
| `test_system.py` | Unit tests |

---

## 🎓 Learning Path

### Beginner
1. Run the quick start steps above
2. Try the web interface
3. Test different comments
4. Explore response templates

### Intermediate
1. Run `example_usage.py` to see programmatic usage
2. Modify response templates
3. Add more training data
4. Experiment with batch processing

### Advanced
1. Try different ML models (SVM, Random Forest)
2. Adjust preprocessing parameters
3. Fine-tune TF-IDF settings
4. Create custom visualizations
5. Integrate with social media APIs

---

## 🚀 Next Steps

After getting familiar with the tool:

1. **Test with Real Data:**
   - Export comments from your social media
   - Format as CSV with a "comment" column
   - Upload to the batch analysis page

2. **Customize Responses:**
   - Edit `response_templates.py`
   - Add your brand voice
   - Create category-specific templates

3. **Improve Accuracy:**
   - Add more examples to `dataset.csv`
   - Retrain the model
   - Compare different model types

4. **Deploy:**
   - Share the tool with your team
   - Set up on a server for team access
   - Integrate with your workflow

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Review code comments in each file
3. Run `python test_system.py` to verify setup
4. Try `python example_usage.py` for usage examples

---

**Ready to get started? Run the 3 commands above and you'll be analyzing comments in minutes! 🎉**

```bash
pip install -r requirements.txt
python model_trainer.py
streamlit run app.py
```
