# Explainable Recommendation System

A production-ready explainable recommendation system that provides clear explanations for why items are recommended to users. This project implements multiple recommendation algorithms with comprehensive evaluation metrics and interactive explanations.

## Features

- **Multiple Recommendation Models**: Popularity-based, Item-based Collaborative Filtering, Matrix Factorization, and Content-based filtering
- **Comprehensive Explanations**: Similarity-based, feature-based, popularity-based, and hybrid explanations
- **Rich Evaluation Metrics**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage, Diversity, and Novelty
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations and explanations
- **Production Ready**: Clean code with type hints, comprehensive testing, and proper documentation

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Explainable-Recommendation-System.git
cd Explainable-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main pipeline:
```bash
python main.py
```

4. Launch the interactive demo:
```bash
streamlit run demo.py
```

## Project Structure

```
explainable-recommendation-system/
├── src/
│   ├── data/
│   │   └── pipeline.py          # Data generation and loading
│   ├── models/
│   │   └── recommenders.py      # Recommendation models
│   ├── evaluation/
│   │   └── metrics.py           # Evaluation metrics
│   ├── explainability/
│   │   └── generator.py         # Explanation generation
│   └── utils/                   # Utility functions
├── configs/
│   └── default.yaml            # Configuration file
├── tests/
│   └── test_recommendation_system.py  # Test suite
├── notebooks/                  # Jupyter notebooks for analysis
├── scripts/                   # Utility scripts
├── assets/                    # Generated visualizations and results
├── data/                      # Data files (generated)
├── main.py                    # Main pipeline script
├── demo.py                    # Streamlit demo
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Usage

### Command Line Interface

The main script provides a comprehensive command-line interface:

```bash
python main.py --help
```

Key options:
- `--data-dir`: Directory for data files (default: "data")
- `--output-dir`: Directory for output files (default: "assets")
- `--seed`: Random seed for reproducibility (default: 42)
- `--n-recommendations`: Number of recommendations to generate (default: 10)
- `--force-regenerate`: Force regenerate synthetic data

### Interactive Demo

The Streamlit demo provides an interactive interface to explore the recommendation system:

```bash
streamlit run demo.py
```

Features:
- User selection and profile visualization
- Model comparison with side-by-side recommendations
- Detailed explanations for each recommendation
- Interactive charts and metrics

### Programmatic Usage

```python
from src.data.pipeline import DataLoader
from src.models.recommenders import ItemBasedCFRecommender
from src.evaluation.metrics import ModelEvaluator, RecommendationMetrics

# Load data
loader = DataLoader()
interactions_df, items_df, users_df = loader.load_data()
train_df, test_df = loader.create_train_test_split(interactions_df)

# Train model
model = ItemBasedCFRecommender()
model.fit(train_df)

# Generate recommendations
user_id = "user_0001"
recommendations = model.recommend(user_id, n_recommendations=10)

# Get explanations
if recommendations:
    item_id = recommendations[0][0]
    explanation = model.explain_recommendation(user_id, item_id)
    print(f"Explanation: {explanation['reason']}")

# Evaluate model
metrics = RecommendationMetrics()
evaluator = ModelEvaluator(metrics)
results = evaluator.evaluate_model(model, test_df, items_df)
print(f"NDCG@10: {results['ndcg@10']:.4f}")
```

## Models

### 1. Popularity Recommender
- **Type**: Baseline model
- **Method**: Recommends items based on average ratings
- **Explanation**: Popularity-based explanations showing item ratings and global averages

### 2. Item-Based Collaborative Filtering
- **Type**: Collaborative filtering
- **Method**: Uses item-item similarity matrix with cosine similarity
- **Explanation**: Similarity-based explanations showing contributing items and similarity scores

### 3. Matrix Factorization
- **Type**: Latent factor model
- **Method**: SVD-based matrix factorization
- **Explanation**: Latent factor explanations showing factor contributions

### 4. Content-Based Recommender
- **Type**: Content-based filtering
- **Method**: Uses item features (category, tags, price, etc.)
- **Explanation**: Feature-based explanations showing feature matches

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Ranking Metrics
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation

### Diversity Metrics
- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Intra-list diversity based on item similarity
- **Novelty**: Average novelty of recommended items

## Explanation Types

### 1. Similarity Explanations
- Shows items similar to user's rated items
- Displays similarity scores and user interaction history
- Highlights which similar items the user has rated

### 2. Feature-Based Explanations
- Shows how item features match user preferences
- Displays feature importance and match scores
- Explains category, price, and tag preferences

### 3. Popularity Explanations
- Shows item popularity scores and trends
- Compares to global and category averages
- Explains why popular items are recommended

### 4. Hybrid Explanations
- Combines multiple explanation types
- Weighted by explanation importance
- Provides comprehensive reasoning

## Configuration

The system uses YAML configuration files. Key settings:

```yaml
# Data settings
data:
  n_users: 1000
  n_items: 500
  n_interactions: 10000
  test_size: 0.2

# Model settings
models:
  item_cf:
    min_similarity: 0.1
  matrix_factorization:
    n_factors: 50

# Evaluation settings
evaluation:
  k_values: [5, 10, 20]
  rating_threshold: 3.0
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

The test suite includes:
- Unit tests for all components
- Integration tests for the complete pipeline
- Data generation and loading tests
- Model training and prediction tests
- Explanation generation tests

## Development

### Code Quality

The project follows Python best practices:
- Type hints throughout
- Google-style docstrings
- PEP 8 compliance (enforced with Black and Ruff)
- Comprehensive test coverage

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

### Adding New Models

To add a new recommendation model:

1. Inherit from `BaseRecommender` in `src/models/recommenders.py`
2. Implement required methods: `fit`, `predict`, `recommend`, `explain_recommendation`
3. Add to model initialization in `main.py`
4. Add tests in `tests/test_recommendation_system.py`

### Adding New Explanation Types

To add new explanation types:

1. Add methods to `ExplanationGenerator` in `src/explainability/generator.py`
2. Update model explanation methods to use new types
3. Add visualization support if needed
4. Add tests for new explanation types

## Performance

The system is optimized for performance:
- Efficient matrix operations with NumPy
- Cached data loading and model training
- Vectorized similarity calculations
- Memory-efficient data structures

## Limitations

- Synthetic data generation (replace with real data for production)
- Simplified user preference modeling
- Basic feature extraction (can be enhanced with deep learning)
- Limited to explicit feedback (can be extended to implicit feedback)

## Future Enhancements

- Deep learning models (Neural Collaborative Filtering, DeepFM)
- Real-time recommendation updates
- Multi-objective optimization (accuracy vs. diversity)
- Advanced explanation methods (attention weights, counterfactuals)
- Production deployment with FastAPI
- Experiment tracking with MLflow/W&B

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{explainable_recommendation_system,
  title={Explainable Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Explainable-Recommendation-System}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.
# Explainable-Recommendation-System
