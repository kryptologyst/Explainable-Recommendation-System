"""Streamlit demo for explainable recommendation system."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.pipeline import DataLoader
from models.recommenders import (
    PopularityRecommender,
    ItemBasedCFRecommender,
    MatrixFactorizationRecommender,
    ContentBasedRecommender
)
from explainability.generator import ExplanationGenerator, ExplanationVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Explainable Recommendation System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .recommendation-item {
        background-color: #fff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data with caching."""
    try:
        loader = DataLoader()
        interactions_df, items_df, users_df = loader.load_data()
        return interactions_df, items_df, users_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_resource
def load_models(items_df):
    """Load and train models with caching."""
    try:
        models = {
            "Popularity": PopularityRecommender(),
            "Item-Based CF": ItemBasedCFRecommender(),
            "Matrix Factorization": MatrixFactorizationRecommender(),
            "Content-Based": ContentBasedRecommender(items_df)
        }
        
        # Load training data
        loader = DataLoader()
        interactions_df, _, _ = loader.load_data()
        train_df, _ = loader.create_train_test_split(interactions_df)
        
        # Train models
        for model in models.values():
            model.fit(train_df)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Explainable Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases an explainable recommendation system that provides clear explanations 
    for why items are recommended to users. The system uses multiple recommendation algorithms 
    and generates different types of explanations including similarity-based, feature-based, 
    and popularity-based explanations.
    """)
    
    # Load data
    with st.spinner("Loading data and models..."):
        interactions_df, items_df, users_df = load_data()
        models = load_models(items_df) if items_df is not None else {}
    
    if interactions_df is None or items_df is None or users_df is None:
        st.error("Failed to load data. Please check the data directory.")
        return
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # User selection
    user_options = ["Select a user..."] + sorted(interactions_df['user_id'].unique().tolist())
    selected_user = st.sidebar.selectbox("Select User", user_options)
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        list(models.keys()),
        default=list(models.keys())[:2] if models else []
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 10)
    
    # Explanation type
    explanation_type = st.sidebar.selectbox(
        "Explanation Type",
        ["All", "Similarity", "Feature-based", "Popularity", "Hybrid"]
    )
    
    if selected_user == "Select a user..." or not selected_models:
        st.info("Please select a user and at least one model from the sidebar.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 User Profile")
        
        # Get user info
        user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
        user_interactions = interactions_df[interactions_df['user_id'] == selected_user]
        
        # Display user info
        st.markdown(f"""
        **User ID:** {selected_user}  
        **Age:** {user_info['age']}  
        **Gender:** {user_info['gender']}  
        **Location:** {user_info['location']}  
        **Preferred Categories:** {user_info['preferred_categories']}  
        **Activity Level:** {user_info['activity_level']}  
        **Total Interactions:** {len(user_interactions)}
        """)
        
        # User's rating history
        if len(user_interactions) > 0:
            st.subheader("📈 Rating History")
            
            # Create rating distribution
            rating_counts = user_interactions['rating'].value_counts().sort_index()
            
            col_hist1, col_hist2 = st.columns(2)
            
            with col_hist1:
                st.bar_chart(rating_counts)
            
            with col_hist2:
                # Top rated items
                top_items = user_interactions.nlargest(5, 'rating')[['item_id', 'rating']]
                st.write("**Top Rated Items:**")
                for _, row in top_items.iterrows():
                    item_name = items_df[items_df['item_id'] == row['item_id']]['title'].iloc[0]
                    st.write(f"• {item_name} ({row['rating']} stars)")
    
    with col2:
        st.header("🎯 Quick Stats")
        
        # Dataset stats
        st.markdown("""
        <div class="metric-card">
            <h4>Dataset Statistics</h4>
            <p><strong>Users:</strong> {:,}</p>
            <p><strong>Items:</strong> {:,}</p>
            <p><strong>Interactions:</strong> {:,}</p>
            <p><strong>Sparsity:</strong> {:.1%}</p>
        </div>
        """.format(
            len(users_df),
            len(items_df),
            len(interactions_df),
            (interactions_df['rating'] == 0).mean()
        ), unsafe_allow_html=True)
    
    # Recommendations section
    st.header("🎯 Recommendations")
    
    if not selected_models:
        st.warning("Please select at least one model from the sidebar.")
        return
    
    # Generate recommendations for each selected model
    recommendations_data = {}
    
    for model_name in selected_models:
        if model_name in models:
            try:
                recommendations = models[model_name].recommend(
                    selected_user, 
                    n_recommendations=n_recommendations
                )
                recommendations_data[model_name] = recommendations
            except Exception as e:
                st.error(f"Error generating recommendations for {model_name}: {e}")
    
    if not recommendations_data:
        st.error("No recommendations could be generated.")
        return
    
    # Display recommendations
    tabs = st.tabs([f"📊 {model_name}" for model_name in recommendations_data.keys()])
    
    for i, (model_name, recommendations) in enumerate(recommendations_data.items()):
        with tabs[i]:
            if not recommendations:
                st.warning(f"No recommendations available for {model_name}")
                continue
            
            st.subheader(f"{model_name} Recommendations")
            
            # Initialize explanation generator
            explainer = ExplanationGenerator(items_df)
            
            for j, (item_id, score) in enumerate(recommendations):
                with st.expander(f"#{j+1}: {item_id} (Score: {score:.3f})", expanded=j<3):
                    
                    # Get item details
                    item_info = items_df[items_df['item_id'] == item_id].iloc[0]
                    
                    col_item1, col_item2 = st.columns([2, 1])
                    
                    with col_item1:
                        st.markdown(f"""
                        **Title:** {item_info['title']}  
                        **Category:** {item_info['category']}  
                        **Price:** ${item_info['price']:.2f}  
                        **Average Rating:** {item_info['rating_avg']:.2f} ⭐  
                        **Tags:** {item_info['tags']}
                        """)
                    
                    with col_item2:
                        st.metric("Recommendation Score", f"{score:.3f}")
                    
                    # Generate explanations
                    st.subheader("🔍 Explanation")
                    
                    try:
                        # Get model explanation
                        model_explanation = models[model_name].explain_recommendation(selected_user, item_id)
                        
                        if model_explanation.get('explanation_type') != 'error':
                            st.markdown(f"""
                            <div class="explanation-box">
                                <strong>Reason:</strong> {model_explanation.get('reason', 'No explanation available')}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show explanation details
                            if 'explanation_details' in model_explanation:
                                st.write("**Details:**")
                                for detail in model_explanation['explanation_details']:
                                    st.write(f"• {detail}")
                            
                            # Additional explanations based on type
                            if explanation_type in ["All", "Similarity"] and model_name == "Item-Based CF":
                                if hasattr(models[model_name], 'item_similarity_matrix'):
                                    item_idx = models[model_name].item_ids.index(item_id)
                                    similarities = models[model_name].item_similarity_matrix[item_idx]
                                    similar_items = []
                                    
                                    for k, sim in enumerate(similarities):
                                        if k != item_idx and sim > 0.1:
                                            similar_items.append((models[model_name].item_ids[k], sim))
                                    
                                    similar_items.sort(key=lambda x: x[1], reverse=True)
                                    
                                    if similar_items:
                                        user_history = user_interactions['item_id'].tolist()
                                        sim_explanation = explainer.generate_similarity_explanation(
                                            item_id, similar_items[:3], user_history
                                        )
                                        
                                        st.write("**Similarity Explanation:**")
                                        st.write(sim_explanation['reason'])
                                        
                                        # Show contributing items
                                        if 'contributing_items' in sim_explanation:
                                            st.write("**Contributing Items:**")
                                            for contrib_item in sim_explanation['contributing_items']:
                                                status = "✅ Rated by you" if contrib_item['user_rated'] else "❓ Not rated"
                                                st.write(f"• {contrib_item['title']} (similarity: {contrib_item['similarity']:.3f}) - {status}")
                            
                            if explanation_type in ["All", "Feature-based"]:
                                user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
                                preferred_cats = user_info['preferred_categories'].split('|')
                                user_preferences = {'category': preferred_cats[0] if preferred_cats else 'Unknown'}
                                
                                feature_explanation = explainer.generate_feature_explanation(
                                    item_id, user_preferences
                                )
                                
                                st.write("**Feature-based Explanation:**")
                                st.write(feature_explanation['reason'])
                                
                                if 'feature_matches' in feature_explanation:
                                    st.write("**Feature Matches:**")
                                    for feature, match_score in feature_explanation['feature_matches'].items():
                                        st.write(f"• {feature}: {match_score:.3f}")
                            
                            if explanation_type in ["All", "Popularity"]:
                                pop_explanation = explainer.generate_popularity_explanation(
                                    item_id, 
                                    item_info['popularity_score'],
                                    items_df['popularity_score'].mean()
                                )
                                
                                st.write("**Popularity Explanation:**")
                                st.write(pop_explanation['reason'])
                        
                        else:
                            st.error(f"Error generating explanation: {model_explanation.get('reason', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")
    
    # Model comparison
    if len(recommendations_data) > 1:
        st.header("📊 Model Comparison")
        
        # Create comparison table
        comparison_data = []
        for model_name, recommendations in recommendations_data.items():
            if recommendations:
                # Calculate some basic metrics
                avg_score = np.mean([score for _, score in recommendations])
                score_std = np.std([score for _, score in recommendations])
                
                comparison_data.append({
                    'Model': model_name,
                    'Avg Score': f"{avg_score:.3f}",
                    'Score Std': f"{score_std:.3f}",
                    'Top Recommendation': recommendations[0][0],
                    'Top Score': f"{recommendations[0][1]:.3f}"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Explainable Recommendation System Demo | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
