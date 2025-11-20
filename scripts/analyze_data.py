#!/usr/bin/env python3
"""Utility script for data analysis and visualization."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.pipeline import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_data_distribution(interactions_df, items_df, users_df, output_dir="assets"):
    """Analyze and visualize data distribution."""
    logger.info("Analyzing data distribution...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Rating distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    rating_counts = interactions_df['rating'].value_counts().sort_index()
    plt.bar(rating_counts.index, rating_counts.values, alpha=0.7)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    
    # 2. User activity distribution
    plt.subplot(2, 3, 2)
    user_activity = interactions_df.groupby('user_id').size()
    plt.hist(user_activity, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Interactions per User')
    plt.ylabel('Frequency')
    plt.title('User Activity Distribution')
    
    # 3. Item popularity distribution
    plt.subplot(2, 3, 3)
    item_popularity = interactions_df.groupby('item_id').size()
    plt.hist(item_popularity, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Interactions per Item')
    plt.ylabel('Frequency')
    plt.title('Item Popularity Distribution')
    
    # 4. Category distribution
    plt.subplot(2, 3, 4)
    category_counts = items_df['category'].value_counts()
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('Item Category Distribution')
    
    # 5. Price distribution
    plt.subplot(2, 3, 5)
    plt.hist(items_df['price'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    
    # 6. Average rating distribution
    plt.subplot(2, 3, 6)
    plt.hist(items_df['rating_avg'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Average Rating')
    plt.ylabel('Frequency')
    plt.title('Average Rating Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path / 'data_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Data distribution analysis saved to {output_path / 'data_distribution_analysis.png'}")


def analyze_user_behavior(interactions_df, users_df, output_dir="assets"):
    """Analyze user behavior patterns."""
    logger.info("Analyzing user behavior...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # User behavior analysis
    user_stats = interactions_df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std'],
        'timestamp': ['min', 'max']
    }).round(3)
    
    user_stats.columns = ['interaction_count', 'avg_rating', 'rating_std', 'first_interaction', 'last_interaction']
    user_stats['activity_span'] = user_stats['last_interaction'] - user_stats['first_interaction']
    
    # Merge with user demographics
    user_analysis = users_df.merge(user_stats, left_on='user_id', right_index=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Activity by age
    axes[0, 0].scatter(user_analysis['age'], user_analysis['interaction_count'], alpha=0.6)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Number of Interactions')
    axes[0, 0].set_title('Activity by Age')
    
    # 2. Activity by gender
    gender_activity = user_analysis.groupby('gender')['interaction_count'].mean()
    axes[0, 1].bar(gender_activity.index, gender_activity.values, alpha=0.7)
    axes[0, 1].set_xlabel('Gender')
    axes[0, 1].set_ylabel('Average Interactions')
    axes[0, 1].set_title('Activity by Gender')
    
    # 3. Activity by location
    location_activity = user_analysis.groupby('location')['interaction_count'].mean()
    axes[0, 2].bar(location_activity.index, location_activity.values, alpha=0.7)
    axes[0, 2].set_xlabel('Location')
    axes[0, 2].set_ylabel('Average Interactions')
    axes[0, 2].set_title('Activity by Location')
    
    # 4. Rating patterns by age
    axes[1, 0].scatter(user_analysis['age'], user_analysis['avg_rating'], alpha=0.6)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Average Rating Given')
    axes[1, 0].set_title('Rating Patterns by Age')
    
    # 5. Activity level distribution
    activity_levels = user_analysis['activity_level'].value_counts()
    axes[1, 1].pie(activity_levels.values, labels=activity_levels.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Activity Level Distribution')
    
    # 6. Interaction count vs rating variance
    axes[1, 2].scatter(user_analysis['interaction_count'], user_analysis['rating_std'], alpha=0.6)
    axes[1, 2].set_xlabel('Number of Interactions')
    axes[1, 2].set_ylabel('Rating Standard Deviation')
    axes[1, 2].set_title('Interaction Count vs Rating Consistency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'user_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save user analysis data
    user_analysis.to_csv(output_path / 'user_behavior_analysis.csv', index=False)
    
    logger.info(f"User behavior analysis saved to {output_path}")


def analyze_item_characteristics(items_df, interactions_df, output_dir="assets"):
    """Analyze item characteristics and popularity."""
    logger.info("Analyzing item characteristics...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Item popularity analysis
    item_popularity = interactions_df.groupby('item_id').agg({
        'rating': ['count', 'mean', 'std'],
        'user_id': 'nunique'
    }).round(3)
    
    item_popularity.columns = ['interaction_count', 'avg_rating', 'rating_std', 'unique_users']
    item_popularity['popularity_score'] = item_popularity['interaction_count'] / item_popularity['interaction_count'].max()
    
    # Merge with item metadata
    item_analysis = items_df.merge(item_popularity, left_on='item_id', right_index=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Price vs popularity
    axes[0, 0].scatter(item_analysis['price'], item_analysis['popularity_score'], alpha=0.6)
    axes[0, 0].set_xlabel('Price')
    axes[0, 0].set_ylabel('Popularity Score')
    axes[0, 0].set_title('Price vs Popularity')
    
    # 2. Average rating vs popularity
    axes[0, 1].scatter(item_analysis['avg_rating'], item_analysis['popularity_score'], alpha=0.6)
    axes[0, 1].set_xlabel('Average Rating')
    axes[0, 1].set_ylabel('Popularity Score')
    axes[0, 1].set_title('Average Rating vs Popularity')
    
    # 3. Category popularity
    category_popularity = item_analysis.groupby('category')['popularity_score'].mean()
    axes[0, 2].bar(category_popularity.index, category_popularity.values, alpha=0.7)
    axes[0, 2].set_xlabel('Category')
    axes[0, 2].set_ylabel('Average Popularity')
    axes[0, 2].set_title('Category Popularity')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Price distribution by category
    item_analysis.boxplot(column='price', by='category', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].set_title('Price Distribution by Category')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Rating distribution by category
    item_analysis.boxplot(column='avg_rating', by='category', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Average Rating')
    axes[1, 1].set_title('Rating Distribution by Category')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Interaction count distribution
    axes[1, 2].hist(item_analysis['interaction_count'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Number of Interactions')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Item Interaction Count Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path / 'item_characteristics_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save item analysis data
    item_analysis.to_csv(output_path / 'item_characteristics_analysis.csv', index=False)
    
    logger.info(f"Item characteristics analysis saved to {output_path}")


def generate_summary_report(interactions_df, items_df, users_df, output_dir="assets"):
    """Generate a comprehensive summary report."""
    logger.info("Generating summary report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculate key statistics
    stats = {
        'dataset_size': {
            'users': len(users_df),
            'items': len(items_df),
            'interactions': len(interactions_df)
        },
        'sparsity': {
            'overall_sparsity': (interactions_df['rating'] == 0).mean(),
            'user_coverage': len(interactions_df['user_id'].unique()) / len(users_df),
            'item_coverage': len(interactions_df['item_id'].unique()) / len(items_df)
        },
        'rating_stats': {
            'mean_rating': interactions_df['rating'].mean(),
            'std_rating': interactions_df['rating'].std(),
            'min_rating': interactions_df['rating'].min(),
            'max_rating': interactions_df['rating'].max()
        },
        'user_stats': {
            'avg_interactions_per_user': interactions_df.groupby('user_id').size().mean(),
            'std_interactions_per_user': interactions_df.groupby('user_id').size().std(),
            'min_interactions_per_user': interactions_df.groupby('user_id').size().min(),
            'max_interactions_per_user': interactions_df.groupby('user_id').size().max()
        },
        'item_stats': {
            'avg_interactions_per_item': interactions_df.groupby('item_id').size().mean(),
            'std_interactions_per_item': interactions_df.groupby('item_id').size().std(),
            'min_interactions_per_item': interactions_df.groupby('item_id').size().min(),
            'max_interactions_per_item': interactions_df.groupby('item_id').size().max()
        }
    }
    
    # Create summary report
    report = f"""
# Dataset Summary Report

## Dataset Overview
- **Users**: {stats['dataset_size']['users']:,}
- **Items**: {stats['dataset_size']['items']:,}
- **Interactions**: {stats['dataset_size']['interactions']:,}

## Sparsity Analysis
- **Overall Sparsity**: {stats['sparsity']['overall_sparsity']:.2%}
- **User Coverage**: {stats['sparsity']['user_coverage']:.2%}
- **Item Coverage**: {stats['sparsity']['item_coverage']:.2%}

## Rating Statistics
- **Mean Rating**: {stats['rating_stats']['mean_rating']:.3f}
- **Standard Deviation**: {stats['rating_stats']['std_rating']:.3f}
- **Rating Range**: {stats['rating_stats']['min_rating']:.1f} - {stats['rating_stats']['max_rating']:.1f}

## User Activity
- **Average Interactions per User**: {stats['user_stats']['avg_interactions_per_user']:.1f}
- **Standard Deviation**: {stats['user_stats']['std_interactions_per_user']:.1f}
- **Range**: {stats['user_stats']['min_interactions_per_user']} - {stats['user_stats']['max_interactions_per_user']}

## Item Popularity
- **Average Interactions per Item**: {stats['item_stats']['avg_interactions_per_item']:.1f}
- **Standard Deviation**: {stats['item_stats']['std_interactions_per_item']:.1f}
- **Range**: {stats['item_stats']['min_interactions_per_item']} - {stats['item_stats']['max_interactions_per_item']}

## Category Distribution
"""
    
    # Add category distribution
    category_counts = items_df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = count / len(items_df) * 100
        report += f"- **{category}**: {count} items ({percentage:.1f}%)\n"
    
    # Save report
    with open(output_path / 'dataset_summary_report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"Summary report saved to {output_path / 'dataset_summary_report.md'}")


def main():
    """Main function for data analysis script."""
    parser = argparse.ArgumentParser(description="Data Analysis and Visualization")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="assets", help="Output directory")
    parser.add_argument("--analysis-type", choices=['all', 'distribution', 'user-behavior', 'item-characteristics', 'summary'], 
                       default='all', help="Type of analysis to perform")
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader(args.data_dir)
    interactions_df, items_df, users_df = loader.load_data()
    
    if args.analysis_type in ['all', 'distribution']:
        analyze_data_distribution(interactions_df, items_df, users_df, args.output_dir)
    
    if args.analysis_type in ['all', 'user-behavior']:
        analyze_user_behavior(interactions_df, users_df, args.output_dir)
    
    if args.analysis_type in ['all', 'item-characteristics']:
        analyze_item_characteristics(items_df, interactions_df, args.output_dir)
    
    if args.analysis_type in ['all', 'summary']:
        generate_summary_report(interactions_df, items_df, users_df, args.output_dir)
    
    logger.info("Data analysis completed!")


if __name__ == "__main__":
    main()
