#!/usr/bin/env python3
"""Main script for explainable recommendation system."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.pipeline import DataLoader, create_interaction_matrix
from models.recommenders import (
    PopularityRecommender,
    ItemBasedCFRecommender,
    MatrixFactorizationRecommender,
    ContentBasedRecommender
)
from evaluation.metrics import RecommendationMetrics, ModelEvaluator
from explainability.generator import ExplanationGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_and_prepare_data(
    data_dir: str = "data",
    force_regenerate: bool = False
) -> tuple:
    """Load and prepare data for training and evaluation.
    
    Args:
        data_dir: Directory containing data files
        force_regenerate: Whether to force regenerate data
        
    Returns:
        Tuple of (interactions_df, items_df, users_df, train_df, test_df)
    """
    logger.info("Loading and preparing data...")
    
    loader = DataLoader(data_dir)
    interactions_df, items_df, users_df = loader.load_data(force_regenerate)
    
    # Create train-test split
    train_df, test_df = loader.create_train_test_split(interactions_df)
    
    logger.info(f"Data loaded: {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
    logger.info(f"Train set: {len(train_df)} interactions")
    logger.info(f"Test set: {len(test_df)} interactions")
    
    return interactions_df, items_df, users_df, train_df, test_df


def initialize_models(items_df: pd.DataFrame) -> Dict[str, any]:
    """Initialize all recommendation models.
    
    Args:
        items_df: Items metadata DataFrame
        
    Returns:
        Dictionary mapping model names to model instances
    """
    logger.info("Initializing recommendation models...")
    
    models = {
        "popularity": PopularityRecommender(),
        "item_cf": ItemBasedCFRecommender(min_similarity=0.1),
        "matrix_factorization": MatrixFactorizationRecommender(n_factors=50),
        "content_based": ContentBasedRecommender(items_df)
    }
    
    logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
    return models


def train_models(models: Dict[str, any], train_df: pd.DataFrame) -> None:
    """Train all recommendation models.
    
    Args:
        models: Dictionary mapping model names to model instances
        train_df: Training interactions DataFrame
    """
    logger.info("Training recommendation models...")
    
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        try:
            model.fit(train_df)
            logger.info(f"Successfully trained {name} model")
        except Exception as e:
            logger.error(f"Error training {name} model: {e}")
            # Remove failed model
            del models[name]


def evaluate_models(
    models: Dict[str, any],
    test_df: pd.DataFrame,
    items_df: pd.DataFrame,
    n_recommendations: int = 10
) -> pd.DataFrame:
    """Evaluate all models and return comparison results.
    
    Args:
        models: Dictionary mapping model names to model instances
        test_df: Test interactions DataFrame
        items_df: Items metadata DataFrame
        n_recommendations: Number of recommendations to generate
        
    Returns:
        DataFrame with model comparison results
    """
    logger.info("Evaluating recommendation models...")
    
    # Initialize metrics and evaluator
    metrics = RecommendationMetrics(k_values=[5, 10, 20])
    evaluator = ModelEvaluator(metrics)
    
    # Compare models
    results_df = evaluator.compare_models(models, test_df, items_df, n_recommendations)
    
    logger.info("Model evaluation completed")
    return results_df


def demonstrate_explanations(
    models: Dict[str, any],
    items_df: pd.DataFrame,
    users_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    n_examples: int = 3
) -> None:
    """Demonstrate explanation capabilities.
    
    Args:
        models: Dictionary mapping model names to model instances
        items_df: Items metadata DataFrame
        users_df: Users metadata DataFrame
        interactions_df: Interactions DataFrame
        n_examples: Number of examples to show
    """
    logger.info("Demonstrating explanation capabilities...")
    
    # Initialize explanation generator
    explainer = ExplanationGenerator(items_df)
    
    # Get sample users
    sample_users = interactions_df['user_id'].unique()[:n_examples]
    
    for user_id in sample_users:
        logger.info(f"\n--- Explanations for User: {user_id} ---")
        
        # Get user's history
        user_history = interactions_df[
            interactions_df['user_id'] == user_id
        ]['item_id'].tolist()
        
        logger.info(f"User history: {user_history[:5]}...")  # Show first 5 items
        
        # Generate recommendations with each model
        for model_name, model in models.items():
            try:
                recommendations = model.recommend(user_id, n_recommendations=3)
                
                if recommendations:
                    item_id, score = recommendations[0]
                    
                    # Get explanation from model
                    explanation = model.explain_recommendation(user_id, item_id)
                    
                    logger.info(f"\n{model_name} Model:")
                    logger.info(f"  Top recommendation: {item_id} (score: {score:.3f})")
                    logger.info(f"  Explanation: {explanation.get('reason', 'No explanation available')}")
                    
                    # Generate additional explanations
                    if model_name == "item_cf" and hasattr(model, 'item_similarity_matrix'):
                        # Get similar items for similarity explanation
                        item_idx = model.item_ids.index(item_id)
                        similarities = model.item_similarity_matrix[item_idx]
                        similar_items = []
                        
                        for i, sim in enumerate(similarities):
                            if i != item_idx and sim > 0.1:
                                similar_items.append((model.item_ids[i], sim))
                        
                        similar_items.sort(key=lambda x: x[1], reverse=True)
                        
                        if similar_items:
                            sim_explanation = explainer.generate_similarity_explanation(
                                item_id, similar_items[:3], user_history
                            )
                            logger.info(f"  Similarity explanation: {sim_explanation['reason']}")
                    
                    # Generate feature-based explanation
                    user_row = users_df[users_df['user_id'] == user_id]
                    if not user_row.empty:
                        preferred_cats = user_row.iloc[0]['preferred_categories'].split('|')
                        user_preferences = {'category': preferred_cats[0] if preferred_cats else 'Unknown'}
                        
                        feature_explanation = explainer.generate_feature_explanation(
                            item_id, user_preferences
                        )
                        logger.info(f"  Feature explanation: {feature_explanation['reason']}")
                
            except Exception as e:
                logger.warning(f"Error generating explanations for {model_name}: {e}")


def create_visualizations(
    results_df: pd.DataFrame,
    models: Dict[str, any],
    items_df: pd.DataFrame,
    output_dir: str = "assets"
) -> None:
    """Create visualization plots.
    
    Args:
        results_df: Model comparison results DataFrame
        models: Dictionary mapping model names to model instances
        items_df: Items metadata DataFrame
        output_dir: Directory to save visualizations
    """
    logger.info("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison bar chart
        plt.figure(figsize=(12, 8))
        
        # Select key metrics for visualization
        key_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'coverage']
        available_metrics = [m for m in key_metrics if m in results_df.columns]
        
        if available_metrics:
            x = np.arange(len(results_df))
            width = 0.2
            
            for i, metric in enumerate(available_metrics):
                plt.bar(x + i * width, results_df[metric], width, 
                       label=metric.replace('@', '@').replace('_', ' ').title())
            
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Comparison - Key Metrics')
            plt.xticks(x + width * (len(available_metrics) - 1) / 2, results_df['model'])
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Metric correlation heatmap
        if len(results_df) > 1:
            plt.figure(figsize=(10, 8))
            
            # Select numeric columns for correlation
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = results_df[numeric_cols].corr()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.3f')
                plt.title('Metric Correlation Matrix')
                plt.tight_layout()
                plt.savefig(output_path / 'metric_correlation.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Category distribution
        plt.figure(figsize=(10, 6))
        category_counts = items_df['category'].value_counts()
        
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        plt.title('Item Category Distribution')
        plt.tight_layout()
        plt.savefig(output_path / 'category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Rating distribution
        plt.figure(figsize=(10, 6))
        plt.hist(items_df['rating_avg'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Average Rating')
        plt.ylabel('Frequency')
        plt.title('Distribution of Item Average Ratings')
        plt.tight_layout()
        plt.savefig(output_path / 'rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
        
    except ImportError:
        logger.warning("Matplotlib/seaborn not available for visualization")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")


def main():
    """Main function to run the explainable recommendation system."""
    parser = argparse.ArgumentParser(description="Explainable Recommendation System")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="assets", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate data")
    parser.add_argument("--n-recommendations", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--n-examples", type=int, default=3, help="Number of explanation examples")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization creation")
    
    args = parser.parse_args()
    
    # Set up seeds
    setup_seeds(args.seed)
    
    try:
        # Load and prepare data
        interactions_df, items_df, users_df, train_df, test_df = load_and_prepare_data(
            args.data_dir, args.force_regenerate
        )
        
        # Initialize models
        models = initialize_models(items_df)
        
        # Train models
        train_models(models, train_df)
        
        # Evaluate models
        results_df = evaluate_models(models, test_df, items_df, args.n_recommendations)
        
        # Print results
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(results_df.round(4).to_string(index=False))
        
        # Demonstrate explanations
        demonstrate_explanations(models, items_df, users_df, interactions_df, args.n_examples)
        
        # Create visualizations
        if not args.skip_visualization:
            create_visualizations(results_df, models, items_df, args.output_dir)
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        results_df.to_csv(output_path / 'model_comparison_results.csv', index=False)
        
        logger.info("Explainable recommendation system completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
