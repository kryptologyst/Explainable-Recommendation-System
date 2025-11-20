"""Enhanced explainability features for recommendation systems."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generate various types of explanations for recommendations."""
    
    def __init__(self, items_df: pd.DataFrame) -> None:
        """Initialize explanation generator.
        
        Args:
            items_df: Items metadata DataFrame
        """
        self.items_df = items_df
        self.item_to_features = self._build_item_features()
        
    def _build_item_features(self) -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
        """Build item feature dictionary.
        
        Returns:
            Dictionary mapping item IDs to their features
        """
        features = {}
        for _, item in self.items_df.iterrows():
            features[item['item_id']] = {
                'title': item['title'],
                'category': item['category'],
                'tags': item['tags'].split('|'),
                'price': item['price'],
                'rating_avg': item['rating_avg'],
                'description': item['description']
            }
        return features
    
    def generate_similarity_explanation(
        self,
        item_id: str,
        similar_items: List[Tuple[str, float]],
        user_history: Optional[List[str]] = None
    ) -> Dict[str, Union[str, List]]:
        """Generate explanation based on item similarity.
        
        Args:
            item_id: Target item ID
            similar_items: List of (item_id, similarity_score) tuples
            user_history: Optional list of user's previously rated items
            
        Returns:
            Dictionary containing similarity-based explanation
        """
        item_features = self.item_to_features.get(item_id, {})
        
        explanation_details = []
        contributing_items = []
        
        for similar_item_id, similarity in similar_items[:3]:  # Top 3 similar items
            similar_features = self.item_to_features.get(similar_item_id, {})
            
            # Check if user has rated this similar item
            user_rated = user_history is not None and similar_item_id in user_history
            
            explanation_details.append(
                f"Similar to {similar_features.get('title', similar_item_id)} "
                f"(similarity: {similarity:.3f})"
            )
            
            contributing_items.append({
                'item_id': similar_item_id,
                'title': similar_features.get('title', similar_item_id),
                'similarity': similarity,
                'user_rated': user_rated,
                'category': similar_features.get('category', 'Unknown')
            })
        
        return {
            "explanation_type": "similarity",
            "reason": f"This item is recommended because it's similar to items you've interacted with",
            "target_item": {
                'item_id': item_id,
                'title': item_features.get('title', item_id),
                'category': item_features.get('category', 'Unknown'),
                'tags': item_features.get('tags', [])
            },
            "contributing_items": contributing_items,
            "explanation_details": explanation_details
        }
    
    def generate_feature_explanation(
        self,
        item_id: str,
        user_preferences: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[str, Dict]]:
        """Generate explanation based on item features.
        
        Args:
            item_id: Target item ID
            user_preferences: Dictionary mapping features to preference scores
            feature_importance: Optional dictionary mapping features to importance scores
            
        Returns:
            Dictionary containing feature-based explanation
        """
        item_features = self.item_to_features.get(item_id, {})
        
        # Calculate feature match scores
        feature_matches = {}
        explanation_details = []
        
        for feature, preference in user_preferences.items():
            if feature == 'category' and item_features.get('category') == preference:
                match_score = 1.0
                feature_matches[feature] = match_score
                explanation_details.append(
                    f"Matches your preference for {feature}: {preference}"
                )
            elif feature == 'price_range':
                item_price = item_features.get('price', 0)
                if preference == 'low' and item_price < 100:
                    match_score = 1.0
                elif preference == 'medium' and 100 <= item_price <= 300:
                    match_score = 1.0
                elif preference == 'high' and item_price > 300:
                    match_score = 1.0
                else:
                    match_score = 0.0
                
                feature_matches[feature] = match_score
                explanation_details.append(
                    f"Price ${item_price:.2f} matches your {preference} price preference"
                )
            elif feature in item_features.get('tags', []):
                match_score = 1.0
                feature_matches[feature] = match_score
                explanation_details.append(
                    f"Has the '{feature}' tag that you like"
                )
        
        return {
            "explanation_type": "feature_match",
            "reason": f"This item matches your preferences based on its features",
            "target_item": {
                'item_id': item_id,
                'title': item_features.get('title', item_id),
                'category': item_features.get('category', 'Unknown'),
                'price': item_features.get('price', 0),
                'tags': item_features.get('tags', [])
            },
            "feature_matches": feature_matches,
            "user_preferences": user_preferences,
            "explanation_details": explanation_details
        }
    
    def generate_popularity_explanation(
        self,
        item_id: str,
        popularity_score: float,
        global_average: float,
        category_average: Optional[float] = None
    ) -> Dict[str, Union[str, float]]:
        """Generate explanation based on item popularity.
        
        Args:
            item_id: Target item ID
            popularity_score: Item's popularity score
            global_average: Global average popularity
            category_average: Optional category average popularity
            
        Returns:
            Dictionary containing popularity-based explanation
        """
        item_features = self.item_to_features.get(item_id, {})
        
        explanation_details = [
            f"This item has a popularity score of {popularity_score:.3f}",
            f"This is {'above' if popularity_score > global_average else 'below'} the global average of {global_average:.3f}"
        ]
        
        if category_average is not None:
            explanation_details.append(
                f"This is {'above' if popularity_score > category_average else 'below'} "
                f"the {item_features.get('category', 'category')} average of {category_average:.3f}"
            )
        
        return {
            "explanation_type": "popularity",
            "reason": f"This item is recommended because it's popular among users",
            "target_item": {
                'item_id': item_id,
                'title': item_features.get('title', item_id),
                'category': item_features.get('category', 'Unknown')
            },
            "popularity_score": popularity_score,
            "global_average": global_average,
            "category_average": category_average,
            "explanation_details": explanation_details
        }
    
    def generate_trending_explanation(
        self,
        item_id: str,
        recent_popularity: float,
        historical_popularity: float,
        trend_direction: str
    ) -> Dict[str, Union[str, float]]:
        """Generate explanation based on trending behavior.
        
        Args:
            item_id: Target item ID
            recent_popularity: Recent popularity score
            historical_popularity: Historical popularity score
            trend_direction: Direction of trend ('increasing', 'decreasing', 'stable')
            
        Returns:
            Dictionary containing trending-based explanation
        """
        item_features = self.item_to_features.get(item_id, {})
        
        trend_explanations = {
            'increasing': "This item is becoming more popular recently",
            'decreasing': "This item was popular but is declining",
            'stable': "This item maintains consistent popularity"
        }
        
        explanation_details = [
            f"Recent popularity: {recent_popularity:.3f}",
            f"Historical popularity: {historical_popularity:.3f}",
            f"Trend: {trend_direction}"
        ]
        
        return {
            "explanation_type": "trending",
            "reason": trend_explanations.get(trend_direction, "This item shows interesting popularity trends"),
            "target_item": {
                'item_id': item_id,
                'title': item_features.get('title', item_id),
                'category': item_features.get('category', 'Unknown')
            },
            "recent_popularity": recent_popularity,
            "historical_popularity": historical_popularity,
            "trend_direction": trend_direction,
            "explanation_details": explanation_details
        }
    
    def generate_hybrid_explanation(
        self,
        item_id: str,
        explanations: List[Dict],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[str, List, Dict]]:
        """Generate hybrid explanation combining multiple explanation types.
        
        Args:
            item_id: Target item ID
            explanations: List of explanation dictionaries
            weights: Optional weights for different explanation types
            
        Returns:
            Dictionary containing hybrid explanation
        """
        if not explanations:
            return {"explanation_type": "error", "reason": "No explanations provided"}
        
        # Default weights if not provided
        if weights is None:
            weights = {exp.get('explanation_type', 'unknown'): 1.0 for exp in explanations}
        
        # Combine explanation details
        combined_details = []
        explanation_types = []
        
        for exp in explanations:
            exp_type = exp.get('explanation_type', 'unknown')
            explanation_types.append(exp_type)
            
            if 'explanation_details' in exp:
                weight = weights.get(exp_type, 1.0)
                for detail in exp['explanation_details']:
                    combined_details.append(f"[{exp_type}] {detail}")
        
        # Generate combined reason
        primary_explanation = max(explanations, key=lambda x: weights.get(x.get('explanation_type', 'unknown'), 1.0))
        primary_reason = primary_explanation.get('reason', 'This item is recommended')
        
        return {
            "explanation_type": "hybrid",
            "reason": f"This item is recommended for multiple reasons: {primary_reason}",
            "target_item": explanations[0].get('target_item', {'item_id': item_id}),
            "explanation_types": explanation_types,
            "explanation_weights": weights,
            "individual_explanations": explanations,
            "explanation_details": combined_details
        }


class ExplanationVisualizer:
    """Visualize explanations for recommendations."""
    
    def __init__(self, items_df: pd.DataFrame) -> None:
        """Initialize explanation visualizer.
        
        Args:
            items_df: Items metadata DataFrame
        """
        self.items_df = items_df
        
    def create_similarity_plot(
        self,
        item_id: str,
        similar_items: List[Tuple[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """Create visualization of item similarities.
        
        Args:
            item_id: Target item ID
            similar_items: List of (item_id, similarity_score) tuples
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get item names
            item_names = []
            similarities = []
            
            for similar_item_id, similarity in similar_items[:10]:  # Top 10
                item_name = self.items_df[
                    self.items_df['item_id'] == similar_item_id
                ]['title'].iloc[0] if len(self.items_df[
                    self.items_df['item_id'] == similar_item_id
                ]) > 0 else similar_item_id
                
                item_names.append(item_name)
                similarities.append(similarity)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(item_names)), similarities, color='skyblue', alpha=0.7)
            
            # Add similarity values on bars
            for bar, sim in zip(bars, similarities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{sim:.3f}', ha='center', va='bottom')
            
            plt.xlabel('Similar Items')
            plt.ylabel('Similarity Score')
            plt.title(f'Similarity Scores for {item_id}')
            plt.xticks(range(len(item_names)), item_names, rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
    
    def create_feature_importance_plot(
        self,
        feature_importance: Dict[str, float],
        save_path: Optional[str] = None
    ) -> None:
        """Create visualization of feature importance.
        
        Args:
            feature_importance: Dictionary mapping features to importance scores
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            features = list(feature_importance.keys())
            importance_scores = list(feature_importance.values())
            
            # Sort by importance
            sorted_data = sorted(zip(features, importance_scores), key=lambda x: x[1], reverse=True)
            features, importance_scores = zip(*sorted_data)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(features)), importance_scores, color='lightcoral', alpha=0.7)
            
            # Add importance values on bars
            for bar, imp in zip(bars, importance_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{imp:.3f}', ha='center', va='bottom')
            
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.title('Feature Importance for Recommendation')
            plt.xticks(range(len(features)), features, rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for visualization")


class ExplanationEvaluator:
    """Evaluate the quality of explanations."""
    
    def __init__(self) -> None:
        """Initialize explanation evaluator."""
        pass
    
    def evaluate_explanation_coherence(
        self,
        explanation: Dict[str, Union[str, List]],
        user_history: List[str],
        item_features: Dict[str, Union[str, List]]
    ) -> Dict[str, float]:
        """Evaluate explanation coherence.
        
        Args:
            explanation: Explanation dictionary
            user_history: User's interaction history
            item_features: Item feature dictionary
            
        Returns:
            Dictionary of coherence metrics
        """
        metrics = {}
        
        # Check if explanation mentions items user has interacted with
        if 'contributing_items' in explanation:
            contributing_items = explanation['contributing_items']
            user_interacted_items = set(user_history)
            
            mentioned_interacted = sum(1 for item in contributing_items 
                                    if item.get('user_rated', False))
            
            metrics['user_relevance'] = mentioned_interacted / len(contributing_items) if contributing_items else 0.0
        
        # Check explanation length (more details = better)
        if 'explanation_details' in explanation:
            metrics['explanation_length'] = len(explanation['explanation_details'])
        
        # Check feature coverage
        if 'feature_matches' in explanation:
            feature_matches = explanation['feature_matches']
            metrics['feature_coverage'] = len(feature_matches) / len(item_features) if item_features else 0.0
        
        return metrics
    
    def evaluate_explanation_diversity(
        self,
        explanations: List[Dict[str, Union[str, List]]]
    ) -> float:
        """Evaluate diversity of explanations.
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Diversity score
        """
        if len(explanations) < 2:
            return 0.0
        
        explanation_types = [exp.get('explanation_type', 'unknown') for exp in explanations]
        unique_types = set(explanation_types)
        
        return len(unique_types) / len(explanations)


if __name__ == "__main__":
    # Example usage
    from data.pipeline import DataLoader
    
    # Load data
    loader = DataLoader()
    interactions_df, items_df, users_df = loader.load_data()
    
    # Initialize explanation generator
    explainer = ExplanationGenerator(items_df)
    
    # Generate different types of explanations
    item_id = items_df['item_id'].iloc[0]
    
    # Similarity explanation
    similar_items = [('item_0001', 0.8), ('item_0002', 0.7), ('item_0003', 0.6)]
    sim_explanation = explainer.generate_similarity_explanation(
        item_id, similar_items, ['item_0001', 'item_0002']
    )
    print("Similarity Explanation:", sim_explanation['reason'])
    
    # Feature explanation
    user_preferences = {'category': 'Electronics', 'price_range': 'medium'}
    feature_explanation = explainer.generate_feature_explanation(
        item_id, user_preferences
    )
    print("Feature Explanation:", feature_explanation['reason'])
    
    # Popularity explanation
    pop_explanation = explainer.generate_popularity_explanation(
        item_id, 0.8, 0.5, 0.7
    )
    print("Popularity Explanation:", pop_explanation['reason'])
