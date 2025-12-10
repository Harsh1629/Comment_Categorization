"""
Streamlit Web Application for Comment Categorization
Interactive UI with visualizations and export functionality
"""

import streamlit as st

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Comment Categorization Tool",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
import json
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("WordCloud not available. Word cloud visualizations will be disabled.")
import matplotlib.pyplot as plt
from model_trainer import CommentClassifier
from response_templates import ResponseTemplates
import os

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .praise { background-color: #d4edda; }
    .support { background-color: #d1ecf1; }
    .criticism { background-color: #fff3cd; }
    .hate { background-color: #f8d7da; }
    .threat { background-color: #721c24; color: white; }
    .emotional { background-color: #e2d9f3; }
    .spam { background-color: #f5f5f5; }
    .question { background-color: #cfe2ff; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'd:/ASSIGNMENT/comment_classifier.pkl'
    if os.path.exists(model_path):
        return CommentClassifier.load_model(model_path)
    else:
        st.error("Model not found! Please train the model first by running model_trainer.py")
        return None


@st.cache_data
def load_dataset():
    """Load the training dataset"""
    return pd.read_csv('d:/ASSIGNMENT/dataset.csv')


def create_category_distribution_chart(df):
    """Create an interactive category distribution chart"""
    category_counts = df['Predicted Category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Comment Category Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig


def create_confidence_chart(df):
    """Create confidence score distribution chart"""
    fig = px.box(
        df,
        x='Predicted Category',
        y='Confidence',
        title="Confidence Score Distribution by Category",
        color='Predicted Category',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxis(tickangle=45)
    
    return fig


def create_wordcloud(texts, category):
    """Create word cloud for a category"""
    if not WORDCLOUD_AVAILABLE:
        return None
    
    text = ' '.join(texts)
    
    if len(text.strip()) == 0:
        return None
    
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {category}', fontsize=16, fontweight='bold')
        
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        return None



def create_summary_metrics(df):
    """Create summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comments", len(df))
    
    with col2:
        avg_confidence = df['Confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col3:
        high_priority = len(df[df['Priority'].isin(['Critical', 'Very High', 'High'])])
        st.metric("High Priority", high_priority)
    
    with col4:
        categories = df['Predicted Category'].nunique()
        st.metric("Categories Found", categories)


def export_to_csv(df, filename):
    """Export results to CSV"""
    return df.to_csv(index=False).encode('utf-8')


def export_to_json(df, filename):
    """Export results to JSON"""
    return df.to_json(orient='records', indent=2).encode('utf-8')


def analyze_single_comment(classifier, comment):
    """Analyze a single comment"""
    prediction = classifier.predict([comment])[0]
    probabilities = classifier.predict_proba([comment])[0]
    confidence = max(probabilities)
    
    # Get all category probabilities
    proba_dict = {cat: prob for cat, prob in zip(classifier.categories, probabilities)}
    
    return prediction, confidence, proba_dict


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">💬 Comment Categorization & Reply Assistant</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "📊 Batch Analysis", "🔍 Single Comment", "📈 Analytics", "📚 Response Guide"]
    )
    
    # Load model
    classifier = load_model()
    if classifier is None:
        st.stop()
    
    rt = ResponseTemplates()
    
    # ==================== HOME PAGE ====================
    if page == "🏠 Home":
        st.write("### Welcome to the Comment Categorization Tool!")
        
        st.write("""
        This AI-powered tool helps you efficiently manage and respond to user comments by:
        - **Categorizing** comments into 8 distinct types
        - **Prioritizing** responses based on urgency
        - **Suggesting** appropriate reply templates
        - **Visualizing** comment patterns and trends
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 📋 Supported Categories")
            categories = [
                "✨ Praise - Positive feedback",
                "💪 Support - Encouragement",
                "💡 Constructive Criticism - Helpful feedback",
                "😡 Hate - Negative/abusive",
                "⚠️ Threat - Serious threats",
                "❤️ Emotional - Heartfelt responses",
                "🚫 Spam - Irrelevant content",
                "❓ Question/Suggestion - Inquiries"
            ]
            for cat in categories:
                st.write(f"- {cat}")
        
        with col2:
            st.write("#### 🚀 Quick Start")
            st.write("""
            1. **Single Comment**: Test individual comments
            2. **Batch Analysis**: Upload CSV/JSON files
            3. **Analytics**: View insights and patterns
            4. **Response Guide**: Get reply templates
            
            Use the sidebar to navigate between features!
            """)
        
        st.write("---")
        
        # Demo section
        st.write("### 🎯 Try a Quick Demo")
        demo_comments = [
            "Amazing work! Loved the animation.",
            "This is trash, quit now.",
            "The animation was okay but the voiceover felt off.",
            "How long did this take to create?"
        ]
        
        selected_demo = st.selectbox("Choose a demo comment:", demo_comments)
        
        if st.button("Analyze Demo Comment"):
            with st.spinner("Analyzing..."):
                prediction, confidence, proba_dict = analyze_single_comment(classifier, selected_demo)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.success(f"**Category:** {prediction}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    response = rt.get_response(prediction, 0)
                    st.write("**Suggested Response:**")
                    st.write(response)
                
                with col2:
                    st.write("**All Probabilities:**")
                    for cat, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"{cat}: {prob:.2%}")
    
    # ==================== BATCH ANALYSIS ====================
    elif page == "📊 Batch Analysis":
        st.write("### 📊 Batch Comment Analysis")
        st.write("Upload a CSV or JSON file with comments, or enter them manually.")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Upload File", "Manual Entry", "Use Sample Data"])
        
        df_comments = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload CSV or JSON file", type=['csv', 'json'])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_comments = pd.read_csv(uploaded_file)
                    else:
                        df_comments = pd.read_json(uploaded_file)
                    
                    st.success(f"Loaded {len(df_comments)} comments!")
                    
                    # Select comment column
                    comment_col = st.selectbox("Select the comment column:", df_comments.columns)
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        elif input_method == "Manual Entry":
            st.write("Enter comments (one per line):")
            comments_text = st.text_area("Comments", height=200)
            
            if comments_text and st.button("Process Comments"):
                comments_list = [line.strip() for line in comments_text.split('\n') if line.strip()]
                df_comments = pd.DataFrame({'comment': comments_list})
                comment_col = 'comment'
        
        else:  # Use Sample Data
            if st.button("Load Sample Dataset"):
                df_comments = load_dataset()
                comment_col = 'comment'
                st.success(f"Loaded {len(df_comments)} sample comments!")
        
        # Process comments
        if df_comments is not None and st.button("🚀 Analyze All Comments", type="primary"):
            with st.spinner("Analyzing comments... This may take a moment."):
                
                # Get predictions
                comments_list = df_comments[comment_col].tolist()
                predictions = classifier.predict(comments_list)
                probabilities = classifier.predict_proba(comments_list)
                confidences = [max(probs) for probs in probabilities]
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Comment': comments_list,
                    'Predicted Category': predictions,
                    'Confidence': confidences
                })
                
                # Add response templates and priority
                results_df['Suggested Response'] = results_df['Predicted Category'].apply(
                    lambda x: rt.get_response(x, 0)
                )
                
                action_guide = rt.get_action_guide()
                results_df['Priority'] = results_df['Predicted Category'].apply(
                    lambda x: action_guide.get(x, {}).get('priority', 'Medium')
                )
                
                # Store in session state
                st.session_state['results_df'] = results_df
                
                st.success("✅ Analysis Complete!")
                
                # Display summary metrics
                create_summary_metrics(results_df)
                
                # Display charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_category_distribution_chart(results_df), 
                                  use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_confidence_chart(results_df), 
                                  use_container_width=True)
                
                # Category filter
                st.write("---")
                st.write("### 📋 Detailed Results")
                
                selected_categories = st.multiselect(
                    "Filter by category:",
                    options=list(results_df['Predicted Category'].unique()),
                    default=list(results_df['Predicted Category'].unique())
                )
                
                filtered_df = results_df[results_df['Predicted Category'].isin(selected_categories)]
                
                # Display results
                st.dataframe(filtered_df, use_container_width=True, height=400)
                
                # Export options
                st.write("---")
                st.write("### 💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = export_to_csv(results_df, 'results.csv')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f'comment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
                
                with col2:
                    json_data = export_to_json(results_df, 'results.json')
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f'comment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        mime='application/json'
                    )
                
                with col3:
                    # Export filtered results
                    if len(filtered_df) < len(results_df):
                        csv_filtered = export_to_csv(filtered_df, 'filtered_results.csv')
                        st.download_button(
                            label="📥 Download Filtered",
                            data=csv_filtered,
                            file_name=f'filtered_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv'
                        )
    
    # ==================== SINGLE COMMENT ====================
    elif page == "🔍 Single Comment":
        st.write("### 🔍 Analyze Single Comment")
        st.write("Test the classifier on individual comments.")
        
        comment_input = st.text_area("Enter a comment:", height=150)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary")
        
        if analyze_btn and comment_input:
            with st.spinner("Analyzing..."):
                prediction, confidence, proba_dict = analyze_single_comment(classifier, comment_input)
                
                # Main result
                st.write("---")
                st.write("### 📊 Analysis Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.metric("Category", prediction)
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Priority
                    action_guide = rt.get_action_guide()
                    priority = action_guide.get(prediction, {}).get('priority', 'Medium')
                    
                    priority_color = {
                        'Critical': '🔴',
                        'Very High': '🟠',
                        'High': '🟡',
                        'Medium': '🟢',
                        'Low': '⚪'
                    }
                    
                    st.write(f"**Priority:** {priority_color.get(priority, '⚪')} {priority}")
                
                with col2:
                    st.write("**Probability Distribution:**")
                    
                    # Create probability chart
                    proba_df = pd.DataFrame({
                        'Category': list(proba_dict.keys()),
                        'Probability': list(proba_dict.values())
                    }).sort_values('Probability', ascending=True)
                    
                    fig = px.bar(proba_df, x='Probability', y='Category', orientation='h',
                               title="Category Probabilities",
                               color='Probability',
                               color_continuous_scale='blues')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Response templates
                st.write("---")
                st.write("### 💬 Suggested Responses")
                
                responses = rt.get_all_responses(prediction)
                
                for i, response in enumerate(responses, 1):
                    with st.expander(f"Response Option {i}"):
                        st.write(response)
                        if st.button(f"Copy Response {i}", key=f"copy_{i}"):
                            st.code(response, language=None)
                
                # Action guide
                st.write("---")
                st.write("### 📋 Action Guide")
                
                guide_info = action_guide.get(prediction, {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Action:** {guide_info.get('action', 'N/A')}")
                    st.write(f"**Engagement:** {guide_info.get('engagement', 'N/A')}")
                
                with col2:
                    st.info(f"**Notes:** {guide_info.get('notes', 'N/A')}")
    
    # ==================== ANALYTICS ====================
    elif page == "📈 Analytics":
        st.write("### 📈 Analytics & Insights")
        
        # Check if we have results
        if 'results_df' in st.session_state:
            results_df = st.session_state['results_df']
            
            # Summary metrics
            create_summary_metrics(results_df)
            
            st.write("---")
            
            # Charts
            tab1, tab2, tab3 = st.tabs(["Distribution", "Word Clouds", "Trends"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_category_distribution_chart(results_df), 
                                  use_container_width=True)
                
                with col2:
                    # Priority distribution
                    priority_counts = results_df['Priority'].value_counts()
                    fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                               title="Priority Distribution",
                               labels={'x': 'Priority', 'y': 'Count'},
                               color=priority_counts.index,
                               color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                st.plotly_chart(create_confidence_chart(results_df), use_container_width=True)
            
            with tab2:
                st.write("### Word Clouds by Category")
                
                categories = results_df['Predicted Category'].unique()
                
                for category in categories:
                    category_comments = results_df[results_df['Predicted Category'] == category]['Comment'].tolist()
                    
                    if len(category_comments) > 0:
                        st.write(f"#### {category}")
                        if WORDCLOUD_AVAILABLE:
                            wordcloud_fig = create_wordcloud(category_comments, category)
                            if wordcloud_fig:
                                st.pyplot(wordcloud_fig)
                            else:
                                st.write("Not enough text to generate word cloud.")
                        else:
                            st.info("Word cloud visualization not available. Install 'wordcloud' package to enable.")
                        st.write("---")
            
            with tab3:
                st.write("### Category Statistics")
                
                # Category summary table
                category_summary = results_df.groupby('Predicted Category').agg({
                    'Comment': 'count',
                    'Confidence': 'mean'
                }).rename(columns={'Comment': 'Count', 'Confidence': 'Avg Confidence'})
                
                category_summary['Avg Confidence'] = category_summary['Avg Confidence'].apply(lambda x: f"{x:.2%}")
                category_summary = category_summary.sort_values('Count', ascending=False)
                
                st.dataframe(category_summary, use_container_width=True)
                
                # Top comments by category
                st.write("### Sample Comments by Category")
                
                selected_cat = st.selectbox("Select category:", categories)
                
                cat_comments = results_df[results_df['Predicted Category'] == selected_cat].head(10)
                
                for idx, row in cat_comments.iterrows():
                    with st.expander(f"{row['Comment'][:50]}... (Confidence: {row['Confidence']:.2%})"):
                        st.write(f"**Comment:** {row['Comment']}")
                        st.write(f"**Suggested Response:** {row['Suggested Response']}")
        
        else:
            st.info("No analysis results yet. Please run a batch analysis first!")
            if st.button("Go to Batch Analysis"):
                st.session_state['page'] = "📊 Batch Analysis"
                st.rerun()
    
    # ==================== RESPONSE GUIDE ====================
    elif page == "📚 Response Guide":
        st.write("### 📚 Response Template Guide")
        st.write("Comprehensive guide for responding to different comment types.")
        
        # Priority order
        st.write("#### 🎯 Response Priority Order")
        priority_order = rt.get_priority_order()
        
        for i, category in enumerate(priority_order, 1):
            st.write(f"{i}. **{category}**")
        
        st.write("---")
        
        # Action guide
        st.write("#### 📋 Category Action Guides")
        
        action_guide = rt.get_action_guide()
        
        for category, info in action_guide.items():
            with st.expander(f"**{category}** - Priority: {info['priority']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Action:** {info['action']}")
                    st.write(f"**Engagement:** {info['engagement']}")
                
                with col2:
                    st.info(f"**Notes:** {info['notes']}")
                
                # Show response templates
                st.write("**Response Templates:**")
                responses = rt.get_all_responses(category)
                for i, response in enumerate(responses, 1):
                    st.write(f"{i}. {response}")
        
        st.write("---")
        
        # Interactive response generator
        st.write("#### ✍️ Interactive Response Generator")
        
        selected_category = st.selectbox("Select category:", list(action_guide.keys()))
        
        responses = rt.get_all_responses(selected_category)
        selected_response = st.radio("Choose a template:", 
                                     [f"Option {i+1}" for i in range(len(responses))])
        
        response_idx = int(selected_response.split()[1]) - 1
        
        st.write("**Selected Response:**")
        st.code(responses[response_idx], language=None)
        
        # Customization
        st.write("**Customize Response:**")
        custom_response = st.text_area("Edit the response:", responses[response_idx], height=100)
        
        if st.button("Copy Customized Response"):
            st.success("Response ready to copy!")
            st.code(custom_response, language=None)


if __name__ == "__main__":
    main()
