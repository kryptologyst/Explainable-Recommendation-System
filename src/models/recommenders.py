"""Recommendation models for explainable recommendation system."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Base class for recommendation models."""
    
    def __init__(self, name: str) -> None:
        """Initialize base recommender.
        
        Args:
            name: Model name
        """
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit the recommendation model.
        
        Args:
            interactions_df: User-item interactions DataFrame
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings for a user.
        
        Args:
            user_id: User identifier
            item_ids: Optional list of item IDs to predict for
            
        Returns:
            Dictionary mapping item IDs to predicted ratings
        """
        pass
    
    @abstractmethod
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, score) tuples
        """
        pass
    
    @abstractmethod
    def explain_recommendation(
        self, 
        user_id: str, 
        item_id: str
    ) -> Dict[str, Union[str, float, List]]:
        """Explain why an item was recommended to a user.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Dictionary containing explanation details
        """
        pass


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender (baseline)."""
    
    def __init__(self) -> None:
        """Initialize popularity recommender."""
        super().__init__("Popularity")
        self.item_popularity: Dict[str, float] = {}
        self.global_mean: float = 0.0
        
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit popularity model.
        
        Args:
            interactions_df: User-item interactions DataFrame
        """
        logger.info("Generating popularity recommender...")
        
        # Calculate item popularity (average rating)
        self.item_popularity = interactions_df.groupby('item_id')['rating'].mean().to_dict()
        self.global_mean = interactions_df['rating'].mean()
        
        self.is_fitted = True
        logger.info(f"Fitted popularity model with {len(self.item_popularity)} items")
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings based on item popularity.
        
        Args:
            user_id: User identifier (not used in popularity model)
            item_ids: Optional list of item IDs to predict for
            
        Returns:
            Dictionary mapping item IDs to predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if item_ids is None:
            item_ids = list(self.item_popularity.keys())
        
        predictions = {}
        for item_id in item_ids:
            predictions[item_id] = self.item_popularity.get(item_id, self.global_mean)
        
        return predictions
    
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate popularity-based recommendations.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, score) tuples
        """
        predictions = self.predict(user_id)
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def explain_recommendation(
        self, 
        user_id: str, 
        item_id: str
    ) -> Dict[str, Union[str, float, List]]:
        """Explain popularity-based recommendation.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Dictionary containing explanation details
        """
        popularity_score = self.item_popularity.get(item_id, self.global_mean)
        
        return {
            "explanation_type": "popularity",
            "reason": f"This item is recommended because it has a high average rating of {popularity_score:.2f}",
            "popularity_score": popularity_score,
            "global_mean": self.global_mean,
            "explanation_details": [
                f"Item {item_id} has an average rating of {popularity_score:.2f}",
                f"This is {'above' if popularity_score > self.global_mean else 'below'} the global average of {self.global_mean:.2f}"
            ]
        }


class ItemBasedCFRecommender(BaseRecommender):
    """Item-based collaborative filtering recommender."""
    
    def __init__(self, min_similarity: float = 0.1) -> None:
        """Initialize item-based CF recommender.
        
        Args:
            min_similarity: Minimum similarity threshold for recommendations
        """
        super().__init__("ItemBasedCF")
        self.min_similarity = min_similarity
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.item_ids: List[str] = []
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_ids: List[str] = []
        
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit item-based collaborative filtering model.
        
        Args:
            interactions_df: User-item interactions DataFrame
        """
        logger.info("Fitting item-based collaborative filtering...")
        
        # Create user-item matrix
        self.user_ids = sorted(interactions_df['user_id'].unique())
        self.item_ids = sorted(interactions_df['item_id'].unique())
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        self.user_item_matrix = np.zeros((len(self.user_ids), len(self.item_ids)))
        
        for _, row in interactions_df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Compute item-item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        self.is_fitted = True
        logger.info(f"Fitted item-based CF with {len(self.item_ids)} items")
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings using item-based collaborative filtering.
        
        Args:
            user_id: User identifier
            item_ids: Optional list of item IDs to predict for
            
        Returns:
            Dictionary mapping item IDs to predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_ids:
            return {}
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix[user_idx]
        
        if item_ids is None:
            item_ids = self.item_ids
        
        predictions = {}
        for item_id in item_ids:
            if item_id not in self.item_ids:
                continue
                
            item_idx = self.item_ids.index(item_id)
            
            # Find similar items that the user has rated
            similarities = self.item_similarity_matrix[item_idx]
            rated_items_mask = user_ratings > 0
            
            if np.sum(rated_items_mask) == 0:
                predictions[item_id] = 0.0
                continue
            
            # Weighted average of similar items
            weighted_sum = np.dot(user_ratings, similarities)
            total_similarity = np.sum(similarities[rated_items_mask])
            
            if total_similarity > 0:
                predictions[item_id] = weighted_sum / total_similarity
            else:
                predictions[item_id] = 0.0
        
        return predictions
    
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate item-based CF recommendations.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, score) tuples
        """
        predictions = self.predict(user_id)
        
        if exclude_rated and user_id in self.user_ids:
            user_idx = self.user_ids.index(user_id)
            user_ratings = self.user_item_matrix[user_idx]
            rated_items = set(self.item_ids[i] for i in range(len(self.item_ids)) if user_ratings[i] > 0)
            predictions = {k: v for k, v in predictions.items() if k not in rated_items}
        
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def explain_recommendation(
        self, 
        user_id: str, 
        item_id: str
    ) -> Dict[str, Union[str, float, List]]:
        """Explain item-based CF recommendation.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Dictionary containing explanation details
        """
        if not self.is_fitted or user_id not in self.user_ids:
            return {"explanation_type": "error", "reason": "User not found"}
        
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        user_ratings = self.user_item_matrix[user_idx]
        similarities = self.item_similarity_matrix[item_idx]
        
        # Find top similar items that user has rated
        rated_items_mask = user_ratings > 0
        if np.sum(rated_items_mask) == 0:
            return {"explanation_type": "error", "reason": "No rated items found"}
        
        # Get top contributing items
        contributing_items = []
        for i, (item_id_other, rating, similarity) in enumerate(zip(self.item_ids, user_ratings, similarities)):
            if rating > 0 and similarity > self.min_similarity:
                contributing_items.append((item_id_other, rating, similarity))
        
        contributing_items.sort(key=lambda x: x[2], reverse=True)
        top_contributors = contributing_items[:5]
        
        explanation_details = []
        for item_id_other, rating, similarity in top_contributors:
            explanation_details.append(
                f"You rated {item_id_other} as {rating} (similarity: {similarity:.3f})"
            )
        
        return {
            "explanation_type": "item_similarity",
            "reason": f"This item is recommended because it's similar to items you've rated highly",
            "top_similar_items": [item[0] for item in top_contributors],
            "similarity_scores": [item[2] for item in top_contributors],
            "explanation_details": explanation_details
        }


class MatrixFactorizationRecommender(BaseRecommender):
    """Matrix factorization recommender using SVD."""
    
    def __init__(self, n_factors: int = 50, random_state: int = 42) -> None:
        """Initialize matrix factorization recommender.
        
        Args:
            n_factors: Number of latent factors
            random_state: Random seed
        """
        super().__init__("MatrixFactorization")
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd_model: Optional[TruncatedSVD] = None
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_ids: List[str] = []
        self.item_ids: List[str] = []
        
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit matrix factorization model.
        
        Args:
            interactions_df: User-item interactions DataFrame
        """
        logger.info(f"Fitting matrix factorization with {self.n_factors} factors...")
        
        # Create user-item matrix
        self.user_ids = sorted(interactions_df['user_id'].unique())
        self.item_ids = sorted(interactions_df['item_id'].unique())
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        self.user_item_matrix = np.zeros((len(self.user_ids), len(self.item_ids)))
        
        for _, row in interactions_df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Fit SVD
        self.svd_model = TruncatedSVD(
            n_components=self.n_factors, 
            random_state=self.random_state
        )
        self.svd_model.fit(self.user_item_matrix)
        
        self.is_fitted = True
        logger.info(f"Fitted matrix factorization model")
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings using matrix factorization.
        
        Args:
            user_id: User identifier
            item_ids: Optional list of item IDs to predict for
            
        Returns:
            Dictionary mapping item IDs to predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_ids:
            return {}
        
        user_idx = self.user_ids.index(user_id)
        
        if item_ids is None:
            item_ids = self.item_ids
        
        # Transform user vector
        user_vector = self.user_item_matrix[user_idx].reshape(1, -1)
        user_factors = self.svd_model.transform(user_vector)[0]
        
        # Transform item matrix
        item_factors = self.svd_model.components_.T
        
        # Compute predictions
        predictions = {}
        for item_id in item_ids:
            if item_id not in self.item_ids:
                continue
                
            item_idx = self.item_ids.index(item_id)
            prediction = np.dot(user_factors, item_factors[item_idx])
            predictions[item_id] = max(0, min(5, prediction))  # Clip to rating range
        
        return predictions
    
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate matrix factorization recommendations.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, score) tuples
        """
        predictions = self.predict(user_id)
        
        if exclude_rated and user_id in self.user_ids:
            user_idx = self.user_ids.index(user_id)
            user_ratings = self.user_item_matrix[user_idx]
            rated_items = set(self.item_ids[i] for i in range(len(self.item_ids)) if user_ratings[i] > 0)
            predictions = {k: v for k, v in predictions.items() if k not in rated_items}
        
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def explain_recommendation(
        self, 
        user_id: str, 
        item_id: str
    ) -> Dict[str, Union[str, float, List]]:
        """Explain matrix factorization recommendation.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Dictionary containing explanation details
        """
        if not self.is_fitted or user_id not in self.user_ids:
            return {"explanation_type": "error", "reason": "User not found"}
        
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        # Get user and item factors
        user_vector = self.user_item_matrix[user_idx].reshape(1, -1)
        user_factors = self.svd_model.transform(user_vector)[0]
        item_factors = self.svd_model.components_[:, item_idx]
        
        # Find most important factors
        factor_contributions = user_factors * item_factors
        top_factors = np.argsort(np.abs(factor_contributions))[-5:][::-1]
        
        explanation_details = []
        for factor_idx in top_factors:
            contribution = factor_contributions[factor_idx]
            explanation_details.append(
                f"Factor {factor_idx}: user={user_factors[factor_idx]:.3f}, "
                f"item={item_factors[factor_idx]:.3f}, contribution={contribution:.3f}"
            )
        
        return {
            "explanation_type": "latent_factors",
            "reason": f"This item matches your preferences in {len(top_factors)} key dimensions",
            "top_factors": top_factors.tolist(),
            "factor_contributions": factor_contributions[top_factors].tolist(),
            "explanation_details": explanation_details
        }


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using item features."""
    
    def __init__(self, items_df: pd.DataFrame) -> None:
        """Initialize content-based recommender.
        
        Args:
            items_df: Items metadata DataFrame
        """
        super().__init__("ContentBased")
        self.items_df = items_df
        self.item_features: Optional[np.ndarray] = None
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.item_ids: List[str] = []
        
    def _extract_features(self) -> np.ndarray:
        """Extract features from item metadata.
        
        Returns:
            Feature matrix
        """
        features = []
        
        for _, item in self.items_df.iterrows():
            feature_vector = []
            
            # Category encoding
            categories = ['Electronics', 'Books', 'Movies', 'Music', 'Clothing', 'Sports', 'Home', 'Food']
            category_vector = [1 if item['category'] == cat else 0 for cat in categories]
            feature_vector.extend(category_vector)
            
            # Price (normalized)
            feature_vector.append(item['price'] / 500.0)  # Normalize to [0, 1]
            
            # Average rating (normalized)
            feature_vector.append(item['rating_avg'] / 5.0)
            
            # Popularity score
            feature_vector.append(item['popularity_score'])
            
            # Tags (simple bag of words)
            tags = item['tags'].split('|')
            all_tags = set()
            for _, other_item in self.items_df.iterrows():
                all_tags.update(other_item['tags'].split('|'))
            
            tag_vector = [1 if tag in tags else 0 for tag in sorted(all_tags)]
            feature_vector.extend(tag_vector)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit content-based model.
        
        Args:
            interactions_df: User-item interactions DataFrame
        """
        logger.info("Fitting content-based recommender...")
        
        self.item_ids = sorted(self.items_df['item_id'].unique())
        self.item_features = self._extract_features()
        
        # Compute item-item similarity
        self.item_similarity_matrix = cosine_similarity(self.item_features)
        
        self.is_fitted = True
        logger.info(f"Fitted content-based model with {len(self.item_ids)} items")
    
    def predict(self, user_id: str, item_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict ratings using content-based filtering.
        
        Args:
            user_id: User identifier
            item_ids: Optional list of item IDs to predict for
            
        Returns:
            Dictionary mapping item IDs to predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if item_ids is None:
            item_ids = self.item_ids
        
        # For content-based, we use item similarity to liked items
        # This is a simplified version - in practice, you'd need user preferences
        predictions = {}
        for item_id in item_ids:
            if item_id not in self.item_ids:
                continue
            
            item_idx = self.item_ids.index(item_id)
            # Use average rating as base score
            item_row = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
            predictions[item_id] = item_row['rating_avg']
        
        return predictions
    
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate content-based recommendations.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, score) tuples
        """
        predictions = self.predict(user_id)
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def explain_recommendation(
        self, 
        user_id: str, 
        item_id: str
    ) -> Dict[str, Union[str, float, List]]:
        """Explain content-based recommendation.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Dictionary containing explanation details
        """
        if not self.is_fitted or item_id not in self.item_ids:
            return {"explanation_type": "error", "reason": "Item not found"}
        
        item_idx = self.item_ids.index(item_id)
        item_row = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
        
        # Find similar items
        similarities = self.item_similarity_matrix[item_idx]
        similar_items = []
        for i, similarity in enumerate(similarities):
            if i != item_idx and similarity > 0.1:
                similar_items.append((self.item_ids[i], similarity))
        
        similar_items.sort(key=lambda x: x[1], reverse=True)
        top_similar = similar_items[:5]
        
        explanation_details = [
            f"Category: {item_row['category']}",
            f"Average rating: {item_row['rating_avg']:.2f}",
            f"Price: ${item_row['price']:.2f}",
            f"Tags: {item_row['tags']}"
        ]
        
        for similar_item_id, similarity in top_similar:
            similar_item = self.items_df[self.items_df['item_id'] == similar_item_id].iloc[0]
            explanation_details.append(
                f"Similar to {similar_item_id} (similarity: {similarity:.3f}) - {similar_item['category']}"
            )
        
        return {
            "explanation_type": "content_features",
            "reason": f"This item matches your preferences based on its content features",
            "item_features": {
                "category": item_row['category'],
                "rating_avg": item_row['rating_avg'],
                "price": item_row['price'],
                "tags": item_row['tags'].split('|')
            },
            "similar_items": [item[0] for item in top_similar],
            "similarity_scores": [item[1] for item in top_similar],
            "explanation_details": explanation_details
        }


if __name__ == "__main__":
    # Example usage
    from data.pipeline import DataLoader
    
    # Load data
    loader = DataLoader()
    interactions_df, items_df, users_df = loader.load_data()
    
    # Test different models
    models = {
        "popularity": PopularityRecommender(),
        "item_cf": ItemBasedCFRecommender(),
        "matrix_factorization": MatrixFactorizationRecommender(),
        "content_based": ContentBasedRecommender(items_df)
    }
    
    # Fit models
    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(interactions_df)
        
        # Test recommendations
        user_id = interactions_df['user_id'].iloc[0]
        recommendations = model.recommend(user_id, n_recommendations=5)
        print(f"Recommendations for {user_id}: {recommendations}")
        
        # Test explanation
        if recommendations:
            item_id = recommendations[0][0]
            explanation = model.explain_recommendation(user_id, item_id)
            print(f"Explanation for {item_id}: {explanation['reason']}")
