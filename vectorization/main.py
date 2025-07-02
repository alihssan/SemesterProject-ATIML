"""
Main script for running TF-IDF vectorization

This script demonstrates how to use the TF-IDF vectorization with different datasets.
"""

import pandas as pd
import numpy as np
import logging
import os
from tf_idf import create_tfidf_vectors, TFIDFVectorizer, generate_tfidf_dataframes

# Set up logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run TF-IDF vectorization.
    """
    logger.info("Starting TF-IDF vectorization pipeline...")
    
    try:
        # === Load Sample Data ===
        logger.info("Loading sample data...")
        
        # Check if we have the AG News dataset
        dataset_path = "../dataset/ag_news_dataset.py"
        if os.path.exists(dataset_path):
            # Import and load AG News dataset
            import sys
            sys.path.append("../dataset")
            from ag_news_dataset import load_ag_news_dataset
            
            df = load_ag_news_dataset()
            texts = df['text'].tolist()
            labels = df['class_name'].tolist()
            
            logger.info(f"Loaded AG News dataset: {len(texts)} samples")
        else:
            # Create sample data for demonstration
            logger.info("Creating sample data for demonstration...")
            sample_texts = [
                "This is a sample text about technology and computers.",
                "Sports news about football and basketball games.",
                "Business article about stocks and market trends.",
                "Technology review of the latest smartphones.",
                "Sports coverage of the Olympic games.",
                "Business analysis of company earnings.",
                "Computer science research on machine learning.",
                "Football match results and player statistics.",
                "Stock market update and investment advice.",
                "Artificial intelligence developments in tech industry."
            ]
            
            sample_labels = [
                "technology", "sports", "business", "technology",
                "sports", "business", "technology", "sports",
                "business", "technology"
            ]
            
            texts = sample_texts
            labels = sample_labels
        
        # === Method 1: Generate Multiple DataFrames with Different N-gram Settings ===
        logger.info("\n" + "="*60)
        logger.info("METHOD 1: Generate Multiple DataFrames with Different N-gram Settings")
        logger.info("="*60)
        
        vector_dataframes = generate_tfidf_dataframes(
            texts=texts,
            labels=labels,
            max_features=2000,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Display information about each dataframe
        for label, df in vector_dataframes.items():
            logger.info(f"\n📊 {label} DataFrame:")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Features: {df.shape[1] - 1}")  # -1 for class_name column
            logger.info(f"  Samples: {df.shape[0]}")
            logger.info(f"  Classes: {df['class_name'].nunique()}")
        
        # === Method 2: Single TF-IDF Vectorization ===
        logger.info("\n" + "="*60)
        logger.info("METHOD 2: Single TF-IDF Vectorization")
        logger.info("="*60)
        
        # Create TF-IDF vectors
        vectors, feature_names = create_tfidf_vectors(
            texts=texts,
            max_features=1000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english',
            return_feature_names=True
        )
        
        logger.info(f"TF-IDF vectors shape: {vectors.shape}")
        logger.info(f"Number of features: {len(feature_names)}")
        logger.info(f"Sample features: {feature_names[:10]}")
        
        # === Method 3: Using TFIDFVectorizer Class ===
        logger.info("\n" + "="*60)
        logger.info("METHOD 3: Using TFIDFVectorizer Class")
        logger.info("="*60)
        
        vectorizer = TFIDFVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        vectors_class = vectorizer.fit_transform(texts)
        feature_names_class = vectorizer.get_feature_names()
        
        logger.info(f"TF-IDF vectors shape (class): {vectors_class.shape}")
        logger.info(f"Number of features (class): {len(feature_names_class)}")
        
        # === Save Results ===
        logger.info("\n" + "="*60)
        logger.info("SAVING RESULTS")
        logger.info("="*60)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Save individual dataframes
        for label, df in vector_dataframes.items():
            filename = f"results/tfidf_{label.lower().replace(' ', '_').replace('+', 'plus')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved: {filename}")
        
        # Save combined dataframe (using the first one as base)
        combined_df = vector_dataframes['Unigrams'].copy()
        for label, df in vector_dataframes.items():
            if label != 'Unigrams':
                # Add features from other dataframes (excluding class_name)
                feature_cols = [col for col in df.columns if col != 'class_name']
                for col in feature_cols:
                    combined_df[col] = df[col]
        
        combined_df.to_csv("results/tfidf_combined_features.csv", index=False)
        logger.info("Saved: results/tfidf_combined_features.csv")
        
        # Save feature names
        feature_names_df = pd.DataFrame({
            'feature_name': feature_names,
            'feature_index': range(len(feature_names))
        })
        feature_names_df.to_csv("results/tfidf_feature_names.csv", index=False)
        
        # Save vectorizer model
        vectorizer.save_model("results/tfidf_vectorizer.pkl")
        
        # === Display Final Statistics ===
        logger.info("\n" + "="*60)
        logger.info("FINAL STATISTICS")
        logger.info("="*60)
        
        logger.info(f"Number of documents: {len(texts)}")
        logger.info(f"Number of classes: {len(set(labels))}")
        logger.info(f"Classes: {list(set(labels))}")
        
        # Show statistics for each dataframe
        for label, df in vector_dataframes.items():
            feature_cols = [col for col in df.columns if col != 'class_name']
            vectors_only = df[feature_cols].values
            sparsity = 1 - np.count_nonzero(vectors_only) / vectors_only.size
            avg_magnitude = np.mean(np.linalg.norm(vectors_only, axis=1))
            
            logger.info(f"\n{label}:")
            logger.info(f"  Features: {len(feature_cols)}")
            logger.info(f"  Sparsity: {sparsity:.4f}")
            logger.info(f"  Avg magnitude: {avg_magnitude:.4f}")
        
        logger.info("\nTF-IDF vectorization completed successfully!")
        logger.info("Results saved to:")
        logger.info("  - results/tfidf_unigrams.csv")
        logger.info("  - results/tfidf_unigrams_plus_bigrams.csv")
        logger.info("  - results/tfidf_bigrams_only.csv")
        logger.info("  - results/tfidf_combined_features.csv")
        logger.info("  - results/tfidf_feature_names.csv")
        logger.info("  - results/tfidf_vectorizer.pkl")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
