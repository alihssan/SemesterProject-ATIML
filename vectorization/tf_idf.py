"""
TF-IDF Vectorization Module

This module provides functions to create TF-IDF vectors from text data.
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic
that reflects how important a word is to a document in a collection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Optional, Tuple
import pickle
import os


class TFIDFVectorizer:
    """
    A wrapper class for TF-IDF vectorization with additional functionality.
    """
    
    def __init__(self, 
                 max_features: Optional[int] = None,
                 min_df: Union[int, float] = 1,
                 max_df: Union[int, float] = 1.0,
                 ngram_range: Tuple[int, int] = (1, 1),
                 stop_words: Optional[Union[str, List[str]]] = 'english',
                 lowercase: bool = True,
                 strip_accents: Optional[str] = 'unicode',
                 analyzer: str = 'word',
                 token_pattern: Optional[str] = None,
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False,
                 norm: Optional[str] = 'l2'):
        """
        Initialize TF-IDF Vectorizer with specified parameters.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for a term to be included
            max_df: Maximum document frequency for a term to be included
            ngram_range: Range of n-grams to extract
            stop_words: Stop words to remove
            lowercase: Whether to convert text to lowercase
            strip_accents: Method to strip accents
            analyzer: Type of analyzer ('word', 'char', 'char_wb')
            token_pattern: Regex pattern for tokenization
            use_idf: Whether to use inverse document frequency
            smooth_idf: Whether to smooth IDF weights
            sublinear_tf: Whether to apply sublinear tf scaling
            norm: Norm to use for normalization
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words,
            lowercase=lowercase,
            strip_accents=strip_accents,
            analyzer=analyzer,
            token_pattern=token_pattern,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            norm=norm
        )
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, texts: Union[List[str], pd.Series, np.ndarray]) -> 'TFIDFVectorizer':
        """
        Fit the TF-IDF vectorizer on the given texts.
        
        Args:
            texts: List of text documents to fit on
            
        Returns:
            Self for method chaining
        """
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return self
    
    def transform(self, texts: Union[List[str], pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts: List of text documents to transform
            
        Returns:
            TF-IDF vectors as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming data")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: Union[List[str], pd.Series, np.ndarray]) -> np.ndarray:
        """
        Fit the vectorizer and transform the texts in one step.
        
        Args:
            texts: List of text documents to fit and transform
            
        Returns:
            TF-IDF vectors as numpy array
        """
        vectors = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return vectors.toarray()
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary).
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self.feature_names.tolist()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> 'TFIDFVectorizer':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        return self


def create_tfidf_vectors(texts: Union[List[str], pd.Series, np.ndarray],
                        max_features: Optional[int] = None,
                        min_df: Union[int, float] = 1,
                        max_df: Union[int, float] = 1.0,
                        ngram_range: Tuple[int, int] = (1, 1),
                        stop_words: Optional[Union[str, List[str]]] = 'english',
                        lowercase: bool = True,
                        strip_accents: Optional[str] = 'unicode',
                        analyzer: str = 'word',
                        token_pattern: Optional[str] = None,
                        use_idf: bool = True,
                        smooth_idf: bool = True,
                        sublinear_tf: bool = False,
                        norm: Optional[str] = 'l2',
                        return_feature_names: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Create TF-IDF vectors from text data.
    
    Args:
        texts: List of text documents to vectorize
        max_features: Maximum number of features to extract
        min_df: Minimum document frequency for a term to be included
        max_df: Maximum document frequency for a term to be included
        ngram_range: Range of n-grams to extract
        stop_words: Stop words to remove
        lowercase: Whether to convert text to lowercase
        strip_accents: How to handle accents
        analyzer: Type of analyzer to use
        token_pattern: Regex pattern for tokenization
        use_idf: Whether to use inverse document frequency
        smooth_idf: Whether to smooth IDF weights
        sublinear_tf: Whether to apply sublinear tf scaling
        norm: Norm to use for normalization
        return_feature_names: Whether to return feature names along with vectors
        
    Returns:
        TF-IDF vectors as numpy array, optionally with feature names
    """
    vectorizer = TFIDFVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=lowercase,
        strip_accents=strip_accents,
        analyzer=analyzer,
        token_pattern=token_pattern,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
        norm=norm
    )
    
    vectors = vectorizer.fit_transform(texts)
    
    if return_feature_names:
        return vectors, vectorizer.get_feature_names()
    else:
        return vectors


def create_tfidf_vectors_with_preprocessing(texts: Union[List[str], pd.Series, np.ndarray],
                                          preprocess_func: Optional[callable] = None,
                                          **tfidf_params) -> np.ndarray:
    """
    Create TF-IDF vectors with optional preprocessing.
    
    Args:
        texts: List of text documents to vectorize
        preprocess_func: Optional preprocessing function to apply to texts
        **tfidf_params: Additional parameters for TF-IDF vectorization
        
    Returns:
        TF-IDF vectors as numpy array
    """
    if preprocess_func is not None:
        processed_texts = [preprocess_func(text) for text in texts]
    else:
        processed_texts = texts
    
    return create_tfidf_vectors(processed_texts, **tfidf_params)


def get_top_features(vectors: np.ndarray, 
                    feature_names: List[str], 
                    top_k: int = 10,
                    document_idx: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    Get the top k features with highest TF-IDF scores.
    
    Args:
        vectors: TF-IDF vectors
        feature_names: List of feature names
        top_k: Number of top features to return
        document_idx: Index of specific document (None for all documents)
        
    Returns:
        List of (feature_name, score) tuples
    """
    if document_idx is not None:
        if document_idx >= vectors.shape[0]:
            raise ValueError(f"Document index {document_idx} out of range")
        doc_vector = vectors[document_idx]
    else:
        # Average across all documents
        doc_vector = np.mean(vectors, axis=0)
    
    # Get indices of top k features
    top_indices = np.argsort(doc_vector)[-top_k:][::-1]
    
    # Return feature names and scores
    return [(feature_names[i], doc_vector[i]) for i in top_indices]


def analyze_tfidf_features(vectors: np.ndarray, 
                          feature_names: List[str],
                          top_k: int = 20) -> pd.DataFrame:
    """
    Analyze TF-IDF features and return statistics.
    
    Args:
        vectors: TF-IDF vectors
        feature_names: List of feature names
        top_k: Number of top features to analyze
        
    Returns:
        DataFrame with feature analysis
    """
    # Calculate statistics
    mean_scores = np.mean(vectors, axis=0)
    std_scores = np.std(vectors, axis=0)
    max_scores = np.max(vectors, axis=0)
    min_scores = np.min(vectors, axis=0)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'feature': feature_names,
        'mean_score': mean_scores,
        'std_score': std_scores,
        'max_score': max_scores,
        'min_score': min_scores,
        'score_range': max_scores - min_scores
    })
    
    # Sort by mean score and return top k
    return analysis_df.sort_values('mean_score', ascending=False).head(top_k)


# Example usage and utility functions
def example_usage():
    """
    Example usage of the TF-IDF vectorization functions.
    """
    # Sample text data
    sample_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning learns through trial and error"
    ]
    
    print("=== TF-IDF Vectorization Example ===")
    
    # Basic usage
    vectors, feature_names = create_tfidf_vectors(
        sample_texts, 
        max_features=50, 
        return_feature_names=True
    )
    
    print(f"Vector shape: {vectors.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Get top features
    top_features = get_top_features(vectors, feature_names, top_k=10)
    print("\nTop 10 features:")
    for feature, score in top_features:
        print(f"  {feature}: {score:.4f}")
    
    # Analyze features
    analysis = analyze_tfidf_features(vectors, feature_names, top_k=10)
    print("\nFeature analysis:")
    print(analysis)
    
    # Using the class-based approach
    print("\n=== Using TFIDFVectorizer Class ===")
    vectorizer = TFIDFVectorizer(max_features=30, ngram_range=(1, 2))
    vectors_class = vectorizer.fit_transform(sample_texts)
    print(f"Vector shape: {vectors_class.shape}")
    
    return vectors, feature_names


if __name__ == "__main__":
    example_usage()
