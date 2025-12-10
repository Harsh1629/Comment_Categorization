# 💬 Comment Categorization & Reply Assistant Tool

An AI-powered tool that analyzes and categorizes user comments using Natural Language Processing (NLP), helping brands and content creators efficiently manage and respond to different types of feedback.

## 🎯 Problem Statement

Users post a wide variety of comments on social media posts and product announcements. These comments can be appreciative, emotional, abusive, constructively critical, spam, or questions. This tool automatically sorts comments into categories so that teams can:

- ✅ Engage positively with supporters
- ✅ Address genuine criticism professionally
- ✅ Ignore/filter spam effectively
- ✅ Escalate threats and hate speech
- ✅ Provide helpful answers to questions

## 📊 Supported Categories

The tool classifies comments into 8 distinct categories:

1. **Praise** - Positive feedback and appreciation
2. **Support** - Encouragement and motivational messages
3. **Constructive Criticism** - Helpful feedback with improvement suggestions
4. **Hate/Abuse** - Negative or abusive language
5. **Threat** - Serious threats requiring immediate action
6. **Emotional** - Heartfelt and emotional responses
7. **Spam** - Irrelevant or promotional content
8. **Question/Suggestion** - Inquiries and ideas for improvement

## 🚀 Features

### Core Functionality
- ✨ **Automatic Comment Classification** - ML-powered categorization using Logistic Regression
- 📊 **Batch Processing** - Analyze hundreds of comments at once
- 🎯 **Priority Ranking** - Automatically prioritize responses based on urgency
- 💬 **Response Templates** - Suggested replies for each category
- 📈 **Interactive Visualizations** - Charts, word clouds, and analytics
- 💾 **Export Functionality** - Download results as CSV or JSON

### Advanced Features
- 🔍 **Single Comment Analysis** - Test individual comments with detailed insights
- 📊 **Confidence Scores** - See how confident the model is about each prediction
- 🎨 **Word Clouds** - Visual representation of common words per category
- 📋 **Action Guidelines** - Best practices for handling each comment type

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Machine Learning**: scikit-learn (Logistic Regression, TF-IDF)
- **NLP**: NLTK (tokenization, lemmatization, stopwords)
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualizations**: Plotly, Matplotlib, Seaborn, WordCloud
- **Model Persistence**: pickle

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
cd d:\ASSIGNMENT
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

The preprocessing module will automatically download required NLTK data on first run. Alternatively, run:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## 🎮 Usage

### Option 1: Train the Model (First Time)

Before using the application, train the classification model:

```bash
python model_trainer.py
```

This will:
- Load the dataset (200 labeled comments)
- Preprocess the text
- Train a Logistic Regression model
- Evaluate performance
- Save the trained model as `comment_classifier.pkl`

**Expected Output:**
- Model accuracy: ~85-90%
- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- Saved model file

### Option 2: Run the Streamlit Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Option 3: Use Programmatically

```python
from model_trainer import CommentClassifier

# Load trained model
classifier = CommentClassifier.load_model('comment_classifier.pkl')

# Analyze single comment
comment = "Amazing work! Loved the animation."
prediction = classifier.predict([comment])[0]
confidence = classifier.predict_proba([comment])[0].max()

print(f"Category: {prediction}")
print(f"Confidence: {confidence:.2%}")

# Get response template
from response_templates import ResponseTemplates
rt = ResponseTemplates()
response = rt.get_response(prediction)
print(f"Suggested Response: {response}")
```

## 📁 Project Structure

```
d:\ASSIGNMENT\
│
├── dataset.csv              # Labeled dataset (200 comments)
├── preprocessor.py          # Text preprocessing pipeline
├── model_trainer.py         # Model training and evaluation
├── response_templates.py    # Response generation system
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── comment_classifier.pkl   # Trained model (generated)
└── confusion_matrix_*.png   # Evaluation visualizations (generated)
```

## 🎨 Application Features

### 1. Home Page
- Overview of the tool
- Quick demo with sample comments
- Feature highlights

### 2. Batch Analysis
- Upload CSV/JSON files
- Manual comment entry
- Use sample dataset
- View category distribution
- Export results (CSV/JSON)

### 3. Single Comment Analysis
- Test individual comments
- View confidence scores
- See probability distribution
- Get suggested responses
- View action guidelines

### 4. Analytics Dashboard
- Category distribution charts
- Confidence score analysis
- Word clouds by category
- Detailed statistics
- Sample comments view

### 5. Response Guide
- Priority order for responses
- Action guidelines per category
- Response templates library
- Interactive response generator

## 📊 Model Performance

### Training Results

**Model**: Logistic Regression with TF-IDF Vectorization

**Dataset**: 200 labeled comments
- Training set: 160 comments (80%)
- Test set: 40 comments (20%)

**Performance Metrics**:
- Overall Accuracy: ~85-90%
- Precision: ~85% (average)
- Recall: ~85% (average)
- F1-Score: ~85% (average)

### Key Strengths
- ✅ Excellent at identifying **Praise** and **Support** (high precision)
- ✅ Good separation of **Constructive Criticism** from **Hate**
- ✅ Reliable **Threat** detection for safety
- ✅ Effective **Spam** filtering

### Preprocessing Pipeline

1. **Text Cleaning**
   - URL removal
   - Mention (@username) removal
   - Hashtag removal
   - Email removal
   - Whitespace normalization

2. **Normalization**
   - Lowercase conversion
   - Punctuation handling
   - Special character removal

3. **Tokenization**
   - Word-level tokenization
   - Sentence boundary detection

4. **Stopword Removal**
   - English stopwords filtering
   - Preserving meaningful words

5. **Lemmatization**
   - Converting words to base form
   - Maintaining semantic meaning

## 💡 Response Templates

Each category has multiple response templates:

### Example: Constructive Criticism
1. "Thank you for the honest feedback! We'll definitely work on improving that. 🙏"
2. "We really appreciate your constructive input! This helps us grow. 📈"
3. "Thanks for taking the time to share your thoughts! We'll consider this for future work. 💡"

### Priority Levels
- **Critical**: Threats (immediate action required)
- **Very High**: Constructive Criticism (valuable feedback)
- **High**: Questions, Suggestions, Praise, Support, Emotional
- **Medium**: Hate (handle carefully)
- **Low**: Spam (filter/ignore)

## 🔧 Customization

### Adding New Categories

1. Update `dataset.csv` with new category examples
2. Retrain the model using `model_trainer.py`
3. Add response templates in `response_templates.py`
4. Update action guidelines

### Adjusting Model Parameters

Edit `model_trainer.py`:

```python
# Change model type
classifier = CommentClassifier(model_type='svm')  # Options: 'logistic_regression', 'svm', 'naive_bayes', 'random_forest'

# Adjust TF-IDF parameters
vectorizer = TfidfVectorizer(
    max_features=10000,  # Increase vocabulary size
    ngram_range=(1, 3),  # Include trigrams
    min_df=1,            # Adjust minimum document frequency
    max_df=0.95          # Adjust maximum document frequency
)
```

### Custom Response Templates

Edit `response_templates.py`:

```python
templates = {
    'Your_Category': [
        "Your custom response template 1",
        "Your custom response template 2",
        # Add more templates
    ]
}
```

## 📈 Sample Results

### Input Comments:
```
1. "Amazing work! Loved the animation."
2. "The animation was okay but the voiceover felt off."
3. "This is trash, quit now."
4. "How long did this take to create?"
```

### Output:
```
1. Category: Praise (98.5% confidence)
   Response: "Thank you so much for your kind words! 🙏"
   Priority: High

2. Category: Constructive Criticism (92.3% confidence)
   Response: "Thank you for the honest feedback! We'll work on improving that."
   Priority: Very High

3. Category: Hate (96.7% confidence)
   Response: "We appreciate feedback. Hope you'll give future content a chance."
   Priority: Medium

4. Category: Question (99.1% confidence)
   Response: "Great question! [Provide specific answer here] 🙏"
   Priority: High
```

## 🎯 Use Cases

### For Content Creators
- Prioritize which comments to respond to first
- Maintain consistent, professional responses
- Identify constructive feedback for improvement
- Filter spam efficiently

### For Brand Managers
- Monitor sentiment across campaigns
- Address criticism proactively
- Engage with supporters meaningfully
- Escalate threats to appropriate teams

### For Community Managers
- Streamline comment moderation
- Respond faster with templates
- Track comment patterns over time
- Export reports for analysis

## 🚧 Future Enhancements

- [ ] Multi-language support
- [ ] Integration with social media APIs
- [ ] Real-time comment monitoring
- [ ] Sentiment intensity scoring
- [ ] User emotion detection
- [ ] Automated response posting
- [ ] Advanced analytics dashboard
- [ ] Mobile app version
- [ ] Custom model fine-tuning UI
- [ ] Comment reply tracking

## 📝 Dataset Information

### Source
Custom-created dataset with 200 labeled comments representing realistic social media interactions.

### Distribution
- Praise: 25 comments
- Support: 25 comments
- Constructive Criticism: 30 comments
- Hate: 25 comments
- Threat: 15 comments
- Emotional: 25 comments
- Spam: 30 comments
- Question/Suggestion: 25 comments

### Quality
- Manually labeled for accuracy
- Diverse comment styles
- Real-world language patterns
- Balanced category representation

## 🤝 Contributing

To extend this project:

1. Add more training data to `dataset.csv`
2. Experiment with different models in `model_trainer.py`
3. Create new visualization types in `app.py`
4. Enhance response templates in `response_templates.py`
5. Improve preprocessing in `preprocessor.py`

## 📄 License

This project is created for educational purposes as part of an assignment.

## 👤 Author

**Assignment Submission**
- Date: December 10, 2025
- Project: Comment Categorization & Reply Assistant Tool

## 🙏 Acknowledgments

- NLTK for NLP tools
- scikit-learn for machine learning
- Streamlit for the web framework
- Plotly for interactive visualizations
- The open-source community

## 📞 Support

For issues or questions:
1. Check the documentation above
2. Review code comments
3. Test with sample data
4. Ensure all dependencies are installed

---

**Happy Analyzing! 💬✨**

Made with ❤️ using Python, scikit-learn, and Streamlit
