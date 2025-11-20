"""Evaluation metrics and model comparison for recommendation systems."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Class for computing recommendation evaluation metrics."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]) -> None:
        """Initialize metrics calculator.
        
        Args:
            k_values: List of k values for precision@k, recall@k, etc.
        """
        self.k_values = k_values
        
    def precision_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        return relevant_in_top_k / k
    
    def recall_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        return relevant_in_top_k / len(relevant_items)
    
    def ndcg_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def map_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate MAP@K (Mean Average Precision).
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def hit_rate_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant items
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score (0 or 1)
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        return 1.0 if len(set(top_k_recs) & set(relevant_items)) > 0 else 0.0
    
    def coverage(
        self, 
        all_recommendations: List[List[str]], 
        all_items: List[str]
    ) -> float:
        """Calculate catalog coverage.
        
        Args:
            all_recommendations: List of recommendation lists for all users
            all_items: List of all available items
            
        Returns:
            Coverage score
        """
        if not all_recommendations:
            return 0.0
        
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / len(all_items)
    
    def diversity(
        self, 
        recommendations: List[str], 
        item_similarity_matrix: np.ndarray,
        item_ids: List[str]
    ) -> float:
        """Calculate intra-list diversity.
        
        Args:
            recommendations: List of recommended item IDs
            item_similarity_matrix: Item-item similarity matrix
            item_ids: List of item IDs corresponding to similarity matrix
            
        Returns:
            Diversity score (1 - average similarity)
        """
        if len(recommendations) < 2:
            return 0.0
        
        # Get indices of recommended items
        item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        rec_indices = [item_to_idx[item] for item in recommendations if item in item_to_idx]
        
        if len(rec_indices) < 2:
            return 0.0
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(rec_indices)):
            for j in range(i + 1, len(rec_indices)):
                sim = item_similarity_matrix[rec_indices[i], rec_indices[j]]
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def novelty(
        self, 
        recommendations: List[str], 
        item_popularity: Dict[str, float]
    ) -> float:
        """Calculate novelty (inverse of popularity).
        
        Args:
            recommendations: List of recommended item IDs
            item_popularity: Dictionary mapping item IDs to popularity scores
            
        Returns:
            Average novelty score
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item in recommendations:
            popularity = item_popularity.get(item, 0.0)
            novelty = 1.0 - popularity  # Assuming popularity is normalized [0, 1]
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def evaluate_user(
        self, 
        recommendations: List[str], 
        relevant_items: List[str],
        item_similarity_matrix: Optional[np.ndarray] = None,
        item_ids: Optional[List[str]] = None,
        item_popularity: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Evaluate recommendations for a single user.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            item_similarity_matrix: Optional item similarity matrix for diversity
            item_ids: Optional item IDs for similarity matrix
            item_popularity: Optional item popularity for novelty
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        for k in self.k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(recommendations, relevant_items, k)
            metrics[f'recall@{k}'] = self.recall_at_k(recommendations, relevant_items, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(recommendations, relevant_items, k)
            metrics[f'map@{k}'] = self.map_at_k(recommendations, relevant_items, k)
            metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(recommendations, relevant_items, k)
        
        # Additional metrics
        if item_similarity_matrix is not None and item_ids is not None:
            metrics['diversity'] = self.diversity(recommendations, item_similarity_matrix, item_ids)
        
        if item_popularity is not None:
            metrics['novelty'] = self.novelty(recommendations, item_popularity)
        
        return metrics


class ModelEvaluator:
    """Class for evaluating recommendation models."""
    
    def __init__(self, metrics: RecommendationMetrics) -> None:
        """Initialize model evaluator.
        
        Args:
            metrics: RecommendationMetrics instance
        """
        self.metrics = metrics
        
    def evaluate_model(
        self,
        model,
        test_df: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None,
        n_recommendations: int = 10,
        rating_threshold: float = 3.0
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model
            test_df: Test interactions DataFrame
            items_df: Optional items DataFrame for additional metrics
            n_recommendations: Number of recommendations to generate
            rating_threshold: Minimum rating to consider item as relevant
            
        Returns:
            Dictionary of average metric scores
        """
        logger.info(f"Evaluating {model.name} model...")
        
        all_user_metrics = []
        all_recommendations = []
        
        # Get item popularity for novelty calculation
        item_popularity = None
        if items_df is not None:
            item_popularity = items_df.set_index('item_id')['popularity_score'].to_dict()
        
        # Get item similarity matrix for diversity calculation
        item_similarity_matrix = None
        item_ids = None
        if hasattr(model, 'item_similarity_matrix') and model.item_similarity_matrix is not None:
            item_similarity_matrix = model.item_similarity_matrix
            item_ids = model.item_ids
        
        # Evaluate each user
        for user_id in test_df['user_id'].unique():
            try:
                # Get user's relevant items (high ratings in test set)
                user_test_items = test_df[
                    (test_df['user_id'] == user_id) & 
                    (test_df['rating'] >= rating_threshold)
                ]['item_id'].tolist()
                
                if not user_test_items:
                    continue
                
                # Generate recommendations
                recommendations = model.recommend(
                    user_id, 
                    n_recommendations=n_recommendations,
                    exclude_rated=True
                )
                
                if not recommendations:
                    continue
                
                rec_item_ids = [item_id for item_id, _ in recommendations]
                all_recommendations.append(rec_item_ids)
                
                # Calculate metrics for this user
                user_metrics = self.metrics.evaluate_user(
                    rec_item_ids,
                    user_test_items,
                    item_similarity_matrix,
                    item_ids,
                    item_popularity
                )
                
                all_user_metrics.append(user_metrics)
                
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        if not all_user_metrics:
            logger.warning("No valid evaluations found")
            return {}
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in all_user_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_user_metrics]
            avg_metrics[metric_name] = np.mean(values)
        
        # Add coverage metric
        if all_recommendations:
            all_items = set()
            if items_df is not None:
                all_items = set(items_df['item_id'].unique())
            elif hasattr(model, 'item_ids'):
                all_items = set(model.item_ids)
            
            if all_items:
                avg_metrics['coverage'] = self.metrics.coverage(all_recommendations, list(all_items))
        
        logger.info(f"Evaluation completed for {len(all_user_metrics)} users")
        return avg_metrics
    
    def compare_models(
        self,
        models: Dict[str, any],
        test_df: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None,
        n_recommendations: int = 10
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances
            test_df: Test interactions DataFrame
            items_df: Optional items DataFrame
            n_recommendations: Number of recommendations to generate
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models...")
        
        results = []
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_df, items_df, n_recommendations)
            
            result = {'model': model_name}
            result.update(metrics)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Sort by primary metric (NDCG@10)
        if 'ndcg@10' in results_df.columns:
            results_df = results_df.sort_values('ndcg@10', ascending=False)
        
        return results_df


def calculate_rating_metrics(
    predictions: Dict[str, float],
    actual_ratings: Dict[str, float]
) -> Dict[str, float]:
    """Calculate rating prediction metrics.
    
    Args:
        predictions: Dictionary mapping item IDs to predicted ratings
        actual_ratings: Dictionary mapping item IDs to actual ratings
        
        Returns:
            Dictionary of rating metrics
    """
    common_items = set(predictions.keys()) & set(actual_ratings.keys())
    
    if not common_items:
        return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
    
    pred_values = [predictions[item] for item in common_items]
    actual_values = [actual_ratings[item] for item in common_items]
    
    mse = mean_squared_error(actual_values, pred_values)
    mae = mean_absolute_error(actual_values, pred_values)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


if __name__ == "__main__":
    # Example usage
    from data.pipeline import DataLoader
    from models.recommenders import PopularityRecommender, ItemBasedCFRecommender
    
    # Load data
    loader = DataLoader()
    interactions_df, items_df, users_df = loader.load_data()
    train_df, test_df = loader.create_train_test_split(interactions_df)
    
    # Initialize metrics
    metrics = RecommendationMetrics(k_values=[5, 10])
    evaluator = ModelEvaluator(metrics)
    
    # Test models
    models = {
        "popularity": PopularityRecommender(),
        "item_cf": ItemBasedCFRecommender()
    }
    
    # Fit models
    for model in models.values():
        model.fit(train_df)
    
    # Compare models
    results_df = evaluator.compare_models(models, test_df, items_df)
    print("\nModel Comparison Results:")
    print(results_df.round(4))
