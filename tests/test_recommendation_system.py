"""Tests for explainable recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pipeline import DataGenerator, DataLoader, create_interaction_matrix
from models.recommenders import (
    PopularityRecommender,
    ItemBasedCFRecommender,
    MatrixFactorizationRecommender,
    ContentBasedRecommender
)
from evaluation.metrics import RecommendationMetrics, ModelEvaluator
from explainability.generator import ExplanationGenerator


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_data_generator_init(self):
        """Test data generator initialization."""
        generator = DataGenerator(n_users=100, n_items=50, n_interactions=500)
        assert generator.n_users == 100
        assert generator.n_items == 50
        assert generator.n_interactions == 500
        assert generator.seed == 42
    
    def test_generate_items(self):
        """Test item generation."""
        generator = DataGenerator(n_items=10)
        items_df = generator.generate_items()
        
        assert len(items_df) == 10
        assert 'item_id' in items_df.columns
        assert 'title' in items_df.columns
        assert 'category' in items_df.columns
        assert 'tags' in items_df.columns
        assert 'price' in items_df.columns
        assert 'rating_avg' in items_df.columns
    
    def test_generate_users(self):
        """Test user generation."""
        generator = DataGenerator(n_users=10)
        users_df = generator.generate_users()
        
        assert len(users_df) == 10
        assert 'user_id' in users_df.columns
        assert 'age' in users_df.columns
        assert 'gender' in users_df.columns
        assert 'location' in users_df.columns
        assert 'preferred_categories' in users_df.columns
    
    def test_generate_interactions(self):
        """Test interaction generation."""
        generator = DataGenerator(n_interactions=100)
        items_df = generator.generate_items()
        users_df = generator.generate_users()
        interactions_df = generator.generate_interactions(items_df, users_df)
        
        assert len(interactions_df) == 100
        assert 'user_id' in interactions_df.columns
        assert 'item_id' in interactions_df.columns
        assert 'rating' in interactions_df.columns
        assert 'timestamp' in interactions_df.columns
        assert 'weight' in interactions_df.columns


class TestRecommenders:
    """Test recommendation models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        generator = DataGenerator(n_users=50, n_items=30, n_interactions=200)
        items_df = generator.generate_items()
        users_df = generator.generate_users()
        interactions_df = generator.generate_interactions(items_df, users_df)
        return interactions_df, items_df, users_df
    
    def test_popularity_recommender(self, sample_data):
        """Test popularity recommender."""
        interactions_df, items_df, users_df = sample_data
        
        model = PopularityRecommender()
        model.fit(interactions_df)
        
        assert model.is_fitted
        assert len(model.item_popularity) > 0
        
        # Test prediction
        user_id = interactions_df['user_id'].iloc[0]
        predictions = model.predict(user_id)
        assert len(predictions) > 0
        
        # Test recommendation
        recommendations = model.recommend(user_id, n_recommendations=5)
        assert len(recommendations) <= 5
        
        # Test explanation
        if recommendations:
            item_id = recommendations[0][0]
            explanation = model.explain_recommendation(user_id, item_id)
            assert 'explanation_type' in explanation
            assert 'reason' in explanation
    
    def test_item_cf_recommender(self, sample_data):
        """Test item-based collaborative filtering."""
        interactions_df, items_df, users_df = sample_data
        
        model = ItemBasedCFRecommender()
        model.fit(interactions_df)
        
        assert model.is_fitted
        assert model.item_similarity_matrix is not None
        
        # Test prediction
        user_id = interactions_df['user_id'].iloc[0]
        predictions = model.predict(user_id)
        assert len(predictions) > 0
        
        # Test recommendation
        recommendations = model.recommend(user_id, n_recommendations=5)
        assert len(recommendations) <= 5
    
    def test_matrix_factorization_recommender(self, sample_data):
        """Test matrix factorization recommender."""
        interactions_df, items_df, users_df = sample_data
        
        model = MatrixFactorizationRecommender(n_factors=10)
        model.fit(interactions_df)
        
        assert model.is_fitted
        assert model.svd_model is not None
        
        # Test prediction
        user_id = interactions_df['user_id'].iloc[0]
        predictions = model.predict(user_id)
        assert len(predictions) > 0
    
    def test_content_based_recommender(self, sample_data):
        """Test content-based recommender."""
        interactions_df, items_df, users_df = sample_data
        
        model = ContentBasedRecommender(items_df)
        model.fit(interactions_df)
        
        assert model.is_fitted
        assert model.item_features is not None
        
        # Test prediction
        user_id = interactions_df['user_id'].iloc[0]
        predictions = model.predict(user_id)
        assert len(predictions) > 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        precision_5 = metrics.precision_at_k(recommendations, relevant_items, 5)
        assert precision_5 == 2/5  # 2 relevant items out of 5 recommendations
        
        precision_3 = metrics.precision_at_k(recommendations, relevant_items, 3)
        assert precision_3 == 2/3  # 2 relevant items out of 3 recommendations
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        recall_5 = metrics.recall_at_k(recommendations, relevant_items, 5)
        assert recall_5 == 2/3  # 2 out of 3 relevant items found
        
        recall_10 = metrics.recall_at_k(recommendations, relevant_items, 10)
        assert recall_10 == 2/3  # Still 2 out of 3 relevant items found
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        ndcg_5 = metrics.ndcg_at_k(recommendations, relevant_items, 5)
        assert 0 <= ndcg_5 <= 1
        
        # Perfect case: all recommendations are relevant
        perfect_recs = ['item1', 'item3', 'item6', 'item7', 'item8']
        perfect_ndcg = metrics.ndcg_at_k(perfect_recs, relevant_items, 5)
        assert perfect_ndcg == 1.0
    
    def test_coverage(self):
        """Test coverage calculation."""
        metrics = RecommendationMetrics()
        
        all_recommendations = [
            ['item1', 'item2', 'item3'],
            ['item2', 'item3', 'item4'],
            ['item1', 'item4', 'item5']
        ]
        all_items = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6']
        
        coverage = metrics.coverage(all_recommendations, all_items)
        assert 0 <= coverage <= 1
        assert coverage == 5/6  # 5 unique items recommended out of 6 total


class TestExplanationGenerator:
    """Test explanation generation."""
    
    @pytest.fixture
    def sample_items_df(self):
        """Create sample items DataFrame."""
        data = {
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Item 1', 'Item 2', 'Item 3'],
            'category': ['Electronics', 'Books', 'Movies'],
            'tags': ['tech|gadget', 'fiction|mystery', 'action|thriller'],
            'price': [100.0, 20.0, 15.0],
            'rating_avg': [4.5, 3.8, 4.2],
            'description': ['Tech item', 'Book item', 'Movie item']
        }
        return pd.DataFrame(data)
    
    def test_explanation_generator_init(self, sample_items_df):
        """Test explanation generator initialization."""
        generator = ExplanationGenerator(sample_items_df)
        assert len(generator.item_to_features) == 3
        assert 'item1' in generator.item_to_features
    
    def test_similarity_explanation(self, sample_items_df):
        """Test similarity explanation generation."""
        generator = ExplanationGenerator(sample_items_df)
        
        similar_items = [('item2', 0.8), ('item3', 0.6)]
        explanation = generator.generate_similarity_explanation(
            'item1', similar_items, ['item2']
        )
        
        assert explanation['explanation_type'] == 'similarity'
        assert 'reason' in explanation
        assert 'contributing_items' in explanation
        assert len(explanation['contributing_items']) == 2
    
    def test_feature_explanation(self, sample_items_df):
        """Test feature explanation generation."""
        generator = ExplanationGenerator(sample_items_df)
        
        user_preferences = {'category': 'Electronics', 'price_range': 'medium'}
        explanation = generator.generate_feature_explanation(
            'item1', user_preferences
        )
        
        assert explanation['explanation_type'] == 'feature_match'
        assert 'reason' in explanation
        assert 'feature_matches' in explanation
    
    def test_popularity_explanation(self, sample_items_df):
        """Test popularity explanation generation."""
        generator = ExplanationGenerator(sample_items_df)
        
        explanation = generator.generate_popularity_explanation(
            'item1', 0.8, 0.5, 0.7
        )
        
        assert explanation['explanation_type'] == 'popularity'
        assert 'reason' in explanation
        assert explanation['popularity_score'] == 0.8
        assert explanation['global_average'] == 0.5


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_data_loader_init(self):
        """Test data loader initialization."""
        loader = DataLoader("test_data")
        assert loader.data_dir == Path("test_data")
    
    @patch('src.data.pipeline.DataGenerator')
    def test_load_data_generation(self, mock_generator):
        """Test data loading with generation."""
        # Mock the generator
        mock_gen_instance = Mock()
        mock_gen_instance.generate_items.return_value = pd.DataFrame({'item_id': ['item1']})
        mock_gen_instance.generate_users.return_value = pd.DataFrame({'user_id': ['user1']})
        mock_gen_instance.generate_interactions.return_value = pd.DataFrame({
            'user_id': ['user1'], 'item_id': ['item1'], 'rating': [5]
        })
        mock_generator.return_value = mock_gen_instance
        
        loader = DataLoader("test_data")
        interactions_df, items_df, users_df = loader.load_data(force_regenerate=True)
        
        assert len(interactions_df) == 1
        assert len(items_df) == 1
        assert len(users_df) == 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline."""
        # Generate data
        generator = DataGenerator(n_users=20, n_items=15, n_interactions=50)
        items_df = generator.generate_items()
        users_df = generator.generate_users()
        interactions_df = generator.generate_interactions(items_df, users_df)
        
        # Create train-test split
        loader = DataLoader()
        train_df, test_df = loader.create_train_test_split(interactions_df)
        
        # Train models
        models = {
            'popularity': PopularityRecommender(),
            'item_cf': ItemBasedCFRecommender()
        }
        
        for model in models.values():
            model.fit(train_df)
        
        # Evaluate models
        metrics = RecommendationMetrics(k_values=[5])
        evaluator = ModelEvaluator(metrics)
        
        results_df = evaluator.compare_models(models, test_df, items_df)
        
        assert len(results_df) == 2
        assert 'model' in results_df.columns
        assert 'precision@5' in results_df.columns
        
        # Test explanations
        explainer = ExplanationGenerator(items_df)
        user_id = train_df['user_id'].iloc[0]
        
        recommendations = models['popularity'].recommend(user_id, n_recommendations=3)
        if recommendations:
            item_id = recommendations[0][0]
            explanation = models['popularity'].explain_recommendation(user_id, item_id)
            assert 'explanation_type' in explanation


if __name__ == "__main__":
    pytest.main([__file__])
