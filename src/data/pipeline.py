"""Data pipeline for explainable recommendation system."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate realistic synthetic recommendation data with explainable patterns."""
    
    def __init__(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        n_interactions: int = 10000,
        seed: int = 42,
    ) -> None:
        """Initialize data generator.
        
        Args:
            n_users: Number of users to generate
            n_items: Number of items to generate
            n_interactions: Number of interactions to generate
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.seed = seed
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_items(self) -> pd.DataFrame:
        """Generate item metadata with explainable features.
        
        Returns:
            DataFrame with item metadata including categories, tags, and text descriptions
        """
        categories = ['Electronics', 'Books', 'Movies', 'Music', 'Clothing', 'Sports', 'Home', 'Food']
        tags_pool = {
            'Electronics': ['gadget', 'tech', 'wireless', 'smart', 'portable'],
            'Books': ['fiction', 'non-fiction', 'mystery', 'romance', 'sci-fi'],
            'Movies': ['action', 'comedy', 'drama', 'thriller', 'horror'],
            'Music': ['rock', 'pop', 'jazz', 'classical', 'electronic'],
            'Clothing': ['casual', 'formal', 'sporty', 'vintage', 'trendy'],
            'Sports': ['outdoor', 'fitness', 'team', 'individual', 'equipment'],
            'Home': ['decor', 'furniture', 'kitchen', 'garden', 'tools'],
            'Food': ['healthy', 'organic', 'spicy', 'sweet', 'savory']
        }
        
        items = []
        for i in range(self.n_items):
            category = np.random.choice(categories)
            tags = np.random.choice(tags_pool[category], size=np.random.randint(2, 5), replace=False)
            
            item = {
                'item_id': f'item_{i:04d}',
                'title': f'{category} Item {i}',
                'category': category,
                'tags': '|'.join(tags),
                'price': np.random.uniform(10, 500),
                'rating_avg': np.random.uniform(2.5, 5.0),
                'description': f'A {category.lower()} item with tags: {", ".join(tags)}',
                'popularity_score': np.random.exponential(1.0)
            }
            items.append(item)
            
        return pd.DataFrame(items)
    
    def generate_users(self) -> pd.DataFrame:
        """Generate user profiles with preferences.
        
        Returns:
            DataFrame with user metadata and preferences
        """
        users = []
        for i in range(self.n_users):
            user = {
                'user_id': f'user_{i:04d}',
                'age': np.random.randint(18, 65),
                'gender': np.random.choice(['M', 'F', 'Other']),
                'location': np.random.choice(['US', 'EU', 'Asia', 'Other']),
                'preferred_categories': '|'.join(np.random.choice(
                    ['Electronics', 'Books', 'Movies', 'Music', 'Clothing', 'Sports', 'Home', 'Food'],
                    size=np.random.randint(2, 5), replace=False
                )),
                'activity_level': np.random.choice(['low', 'medium', 'high'])
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def generate_interactions(
        self, 
        items_df: pd.DataFrame, 
        users_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate realistic user-item interactions with explainable patterns.
        
        Args:
            items_df: Item metadata DataFrame
            users_df: User metadata DataFrame
            
        Returns:
            DataFrame with user-item interactions
        """
        interactions = []
        
        # Create popularity bias
        item_popularity = items_df['popularity_score'].values
        item_popularity = item_popularity / item_popularity.sum()
        
        # Create category preferences for users
        user_preferences = {}
        for _, user in users_df.iterrows():
            preferred_cats = user['preferred_categories'].split('|')
            user_preferences[user['user_id']] = preferred_cats
        
        # Generate interactions
        for _ in range(self.n_interactions):
            user_id = np.random.choice(users_df['user_id'])
            user_cats = user_preferences[user_id]
            
            # Bias item selection towards user's preferred categories
            preferred_items = items_df[items_df['category'].isin(user_cats)]
            if len(preferred_items) > 0 and np.random.random() < 0.7:
                item_id = np.random.choice(preferred_items['item_id'])
            else:
                item_id = np.random.choice(items_df['item_id'], p=item_popularity)
            
            # Generate rating based on item quality and some randomness
            item_row = items_df[items_df['item_id'] == item_id].iloc[0]
            base_rating = item_row['rating_avg']
            rating = max(1, min(5, int(np.random.normal(base_rating, 0.8))))
            
            # Add timestamp with some temporal patterns
            timestamp = np.random.randint(1600000000, 1700000000)  # 2020-2023
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp,
                'weight': 1.0
            })
        
        return pd.DataFrame(interactions)
    
    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset.
        
        Returns:
            Tuple of (interactions_df, items_df, users_df)
        """
        logger.info("Generating synthetic recommendation dataset...")
        
        items_df = self.generate_items()
        users_df = self.generate_users()
        interactions_df = self.generate_interactions(items_df, users_df)
        
        logger.info(f"Generated {len(interactions_df)} interactions for {len(users_df)} users and {len(items_df)} items")
        
        return interactions_df, items_df, users_df


class DataLoader:
    """Load and preprocess recommendation data."""
    
    def __init__(self, data_dir: Union[str, Path] = "data") -> None:
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_data(
        self, 
        force_regenerate: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load or generate data.
        
        Args:
            force_regenerate: Whether to regenerate data even if files exist
            
        Returns:
            Tuple of (interactions_df, items_df, users_df)
        """
        interactions_path = self.data_dir / "interactions.csv"
        items_path = self.data_dir / "items.csv"
        users_path = self.data_dir / "users.csv"
        
        if force_regenerate or not all([interactions_path.exists(), items_path.exists(), users_path.exists()]):
            logger.info("Generating new dataset...")
            generator = DataGenerator()
            interactions_df, items_df, users_df = generator.generate_all_data()
            
            # Save data
            interactions_df.to_csv(interactions_path, index=False)
            items_df.to_csv(items_path, index=False)
            users_df.to_csv(users_path, index=False)
            logger.info(f"Data saved to {self.data_dir}")
        else:
            logger.info("Loading existing dataset...")
            interactions_df = pd.read_csv(interactions_path)
            items_df = pd.read_csv(items_path)
            users_df = pd.read_csv(users_path)
        
        return interactions_df, items_df, users_df
    
    def create_train_test_split(
        self,
        interactions_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create temporal train-test split.
        
        Args:
            interactions_df: Interactions DataFrame
            test_size: Fraction of data to use for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Sort by timestamp for temporal split
        interactions_df = interactions_df.sort_values('timestamp')
        
        # Use last interactions for test set
        n_test = int(len(interactions_df) * test_size)
        train_df = interactions_df.iloc[:-n_test].copy()
        test_df = interactions_df.iloc[-n_test:].copy()
        
        logger.info(f"Train set: {len(train_df)} interactions")
        logger.info(f"Test set: {len(test_df)} interactions")
        
        return train_df, test_df
    
    def create_user_based_split(
        self,
        interactions_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create user-based train-test split (leave-last-k per user).
        
        Args:
            interactions_df: Interactions DataFrame
            test_size: Fraction of interactions per user to use for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_interactions = []
        test_interactions = []
        
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id].sort_values('timestamp')
            n_user_interactions = len(user_interactions)
            n_test_user = max(1, int(n_user_interactions * test_size))
            
            train_interactions.append(user_interactions.iloc[:-n_test_user])
            test_interactions.append(user_interactions.iloc[-n_test_user:])
        
        train_df = pd.concat(train_interactions, ignore_index=True)
        test_df = pd.concat(test_interactions, ignore_index=True)
        
        logger.info(f"User-based split - Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df


def create_interaction_matrix(
    interactions_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None,
    items_df: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Create user-item interaction matrix.
    
    Args:
        interactions_df: Interactions DataFrame
        users_df: Optional users DataFrame for ordering
        items_df: Optional items DataFrame for ordering
        
    Returns:
        Tuple of (interaction_matrix, user_ids, item_ids)
    """
    if users_df is not None:
        user_ids = sorted(users_df['user_id'].unique())
    else:
        user_ids = sorted(interactions_df['user_id'].unique())
    
    if items_df is not None:
        item_ids = sorted(items_df['item_id'].unique())
    else:
        item_ids = sorted(interactions_df['item_id'].unique())
    
    # Create mapping
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    # Create matrix
    matrix = np.zeros((len(user_ids), len(item_ids)))
    
    for _, row in interactions_df.iterrows():
        user_idx = user_to_idx[row['user_id']]
        item_idx = item_to_idx[row['item_id']]
        matrix[user_idx, item_idx] = row['rating']
    
    return matrix, user_ids, item_ids


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    interactions_df, items_df, users_df = loader.load_data()
    
    print("Dataset Overview:")
    print(f"Users: {len(users_df)}")
    print(f"Items: {len(items_df)}")
    print(f"Interactions: {len(interactions_df)}")
    print(f"Average rating: {interactions_df['rating'].mean():.2f}")
    print(f"Sparsity: {(interactions_df['rating'] == 0).mean():.2%}")
    
    # Create train-test split
    train_df, test_df = loader.create_train_test_split(interactions_df)
    
    # Create interaction matrix
    matrix, user_ids, item_ids = create_interaction_matrix(train_df, users_df, items_df)
    print(f"Interaction matrix shape: {matrix.shape}")
