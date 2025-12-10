"""
Example Usage Script
Demonstrates how to use the Comment Classifier programmatically
"""

from model_trainer import CommentClassifier
from response_templates import ResponseTemplates
import pandas as pd


def main():
    """
    Main example demonstrating the comment classification workflow
    """
    
    print("="*70)
    print("Comment Categorization & Reply Assistant - Example Usage")
    print("="*70)
    
    # Example comments to analyze
    example_comments = [
        "Amazing work! Loved the animation.",
        "This is trash, quit now.",
        "The animation was okay but the voiceover felt off.",
        "How long did this take to create?",
        "This reminded me of my childhood.",
        "Follow me for followers!",
        "Keep going, you're doing great!",
        "I'll report you if this continues.",
        "Can you make one on topic X?",
        "You should add subtitles for accessibility."
    ]
    
    print("\n📊 Loading trained model...")
    try:
        classifier = CommentClassifier.load_model('d:/ASSIGNMENT/comment_classifier.pkl')
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model not found! Please run model_trainer.py first to train the model.")
        print("\nCommand: python model_trainer.py")
        return
    
    print("\n" + "="*70)
    print("Analyzing Comments")
    print("="*70)
    
    # Initialize response template generator
    rt = ResponseTemplates()
    
    # Analyze each comment
    results = []
    
    for i, comment in enumerate(example_comments, 1):
        print(f"\n{'─'*70}")
        print(f"Comment #{i}")
        print(f"{'─'*70}")
        print(f"📝 Text: {comment}")
        
        # Get prediction
        prediction = classifier.predict([comment])[0]
        probabilities = classifier.predict_proba([comment])[0]
        confidence = max(probabilities)
        
        # Get response template
        response = rt.get_response(prediction, 0)
        
        # Get action guide
        action_guide = rt.get_action_guide()
        priority = action_guide.get(prediction, {}).get('priority', 'Medium')
        action = action_guide.get(prediction, {}).get('action', 'N/A')
        
        print(f"\n🎯 Category: {prediction}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"⚡ Priority: {priority}")
        print(f"📋 Action: {action}")
        print(f"💬 Suggested Response:\n   {response}")
        
        # Store results
        results.append({
            'Comment': comment,
            'Category': prediction,
            'Confidence': f"{confidence:.2%}",
            'Priority': priority,
            'Response': response
        })
    
    # Create results dataframe
    print("\n" + "="*70)
    print("Summary Results")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Category distribution
    print("\n" + "="*70)
    print("Category Distribution")
    print("="*70)
    
    category_counts = results_df['Category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    # Export results
    print("\n" + "="*70)
    print("Exporting Results")
    print("="*70)
    
    output_file = 'd:/ASSIGNMENT/example_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✅ Results exported to: {output_file}")
    
    # Show response priority order
    print("\n" + "="*70)
    print("Response Priority Order")
    print("="*70)
    
    priority_order = rt.get_priority_order()
    print("\nRecommended order for responding to comments:")
    for i, category in enumerate(priority_order, 1):
        guide = action_guide.get(category, {})
        print(f"{i}. {category} - {guide.get('priority', 'N/A')} Priority")
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70)
    print("\n💡 Tips:")
    print("  - Run 'streamlit run app.py' for the interactive web interface")
    print("  - Check README.md for detailed documentation")
    print("  - Modify response_templates.py to customize responses")
    print("  - Add more training data to dataset.csv for better accuracy")
    
    print("\n🚀 Ready to process your comments!")


if __name__ == "__main__":
    main()
