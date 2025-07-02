"""
TF-IDF Vectorization Module

This module provides functions to create TF-IDF vectors from text data.
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic
that reflects how important a word is to a document in a collection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, LabelEncoder
from typing import Union, List, Optional, Tuple, Callable, Dict
import pickle
import os
import plotly.express as px


class TFIDFVectorizer:
    """
    A wrapper class for TF-IDF vectorization with additional functionality.
    Supports both raw and pre-tokenized (preprocessed) text.
    """

    def __init__(self,
                 preprocessed: bool = False,
                 **tfidf_params):
        self.preprocessed = preprocessed

        if preprocessed:
            self.vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None,
                **tfidf_params
            )
        else:
            self.vectorizer = TfidfVectorizer(**tfidf_params)

        self.is_fitted = False
        self.feature_names = None

    def fit(self, texts: Union[List[str], List[List[str]]]) -> 'TFIDFVectorizer':
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return self

    def transform(self, texts: Union[List[str], List[List[str]]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming data")
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: Union[List[str], List[List[str]]]) -> np.ndarray:
        vectors = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return vectors.toarray()

    def get_feature_names(self) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self.feature_names.tolist()

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before saving")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'preprocessed': self.preprocessed
            }, f)

    def load_model(self, filepath: str) -> 'TFIDFVectorizer':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        self.preprocessed = data.get('preprocessed', False)
        return self

    def visualize_embeddings(self,
                              vectors: np.ndarray,
                              labels: List[str],
                              method: str = 'pca',
                              n_components: int = 2,
                              title: str = "TF-IDF Embeddings",
                              ngram_label: str = "(1, 1)",
                              selected_classes: Optional[List[str]] = None,
                              save_dir: Optional[str] = None) -> None:
        """
        Visualize the TF-IDF vectors using PCA, t-SNE, or Plotly 3D.

        Args:
            vectors: The TF-IDF vector matrix.
            labels: Class labels for coloring.
            method: 'pca', 'tsne', or '3d' for dimensionality reduction.
            n_components: Number of dimensions (2 or 3 recommended).
            title: Plot title.
            ngram_label: N-gram range label to annotate the plot.
            selected_classes: Optional subset of class names to visualize.
        """
        df = pd.DataFrame(vectors)
        df['label'] = labels

        if selected_classes:
            df = df[df['label'].isin(selected_classes)]

        labels_filtered = df['label'].tolist()
        X_filtered = df.drop(columns='label').values

        # Normalize
        X_filtered = normalize(X_filtered)

        if method == '3d':
            if n_components != 3:
                raise ValueError("For 3D visualization, n_components must be 3")
            reducer = PCA(n_components=3)
            X_reduced = reducer.fit_transform(X_filtered)
            fig = px.scatter_3d(
                x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                color=labels_filtered,
                title=f"3D PCA TF-IDF Visualization {ngram_label} | {title}",
                labels={'color': 'Class'}
            )
            fig.show()
            return

        # PCA before t-SNE for stability if needed
        if method == 'tsne':
            X_pca = PCA(n_components=min(50, X_filtered.shape[1])).fit_transform(X_filtered)
            reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
            X_reduced = reducer.fit_transform(X_pca)
        else:
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X_filtered)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=X_reduced[:, 0], y=X_reduced[:, 1],
            hue=labels_filtered, palette='tab20', legend='full', s=25
        )
        plt.title(f"{method.upper()} Visualization of TF-IDF Features {ngram_label}\n{title}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=2, fontsize='small')
        plt.tight_layout()
        
        # Save plot if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"tfidf_{method}_{ngram_label.replace('(', '').replace(')', '').replace(',', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved TF-IDF visualization to: {filepath}")
        
        plt.show()


def create_tfidf_vectors(texts: Union[List[str], pd.Series, np.ndarray],
                        max_features: int = 5000,
                        min_df: int = 2,
                        max_df: float = 0.95,
                        ngram_range: Tuple[int, int] = (1, 2),
                        stop_words: Union[str, List[str]] = 'english',
                        use_idf: bool = True,
                        smooth_idf: bool = True,
                        sublinear_tf: bool = False,
                        token_pattern: str = r'(?u)\b\w\w+\b',
                        return_feature_names: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Create TF-IDF vectors from text data.
    
    Args:
        texts: List of text documents to vectorize
        max_features: Maximum number of features to extract
        min_df: Minimum document frequency for a term to be included
        max_df: Maximum document frequency for a term to be included
        ngram_range: Range of n-grams to extract
        stop_words: Stop words to remove ('english' or list of words)
        use_idf: Whether to use inverse document frequency
        smooth_idf: Whether to smooth IDF weights
        sublinear_tf: Whether to apply sublinear tf scaling
        token_pattern: Regular expression for token extraction
        return_feature_names: Whether to return feature names along with vectors
        
    Returns:
        TF-IDF vectors as numpy array, optionally with feature names
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TFIDFVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
        token_pattern=token_pattern
    )
    
    # Fit and transform the texts
    vectors = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names()
    
    if return_feature_names:
        return vectors, feature_names
    else:
        return vectors


def generate_tfidf_dataframes(texts: Union[List[str], pd.Series, np.ndarray],
                             labels: Union[List[str], pd.Series, np.ndarray],
                             max_features: int = 2000,
                             min_df: int = 2,
                             max_df: float = 0.95,
                             stop_words: Union[str, List[str]] = 'english',
                             use_idf: bool = True,
                             smooth_idf: bool = True,
                             sublinear_tf: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Generate TF-IDF dataframes with different n-gram settings.
    
    Args:
        texts: List of text documents to vectorize
        labels: List of class labels corresponding to texts
        max_features: Maximum number of features to extract
        min_df: Minimum document frequency for a term to be included
        max_df: Maximum document frequency for a term to be included
        stop_words: Stop words to remove ('english' or list of words)
        use_idf: Whether to use inverse document frequency
        smooth_idf: Whether to smooth IDF weights
        sublinear_tf: Whether to apply sublinear tf scaling
        
    Returns:
        Dictionary containing dataframes for each n-gram setting
    """
    
    # Define n-gram settings
    ngram_settings = [
        ((1, 1), 'Unigrams'),
        ((1, 2), 'Unigrams + Bigrams'),
        ((2, 2), 'Bigrams Only')
    ]
    
    vector_storage = {}
    
    print("📊 Generating TF-IDF DataFrames")
    
    for ngram_range, label in ngram_settings:
        print(f"\n📎 {label}")
        
        # Create TF-IDF vectors
        vectors, feature_names = create_tfidf_vectors(
            texts=texts,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            return_feature_names=True
        )
        
        # Create DataFrame with vectors
        df_vectors = pd.DataFrame(vectors, columns=[f"{label}_f{i}" for i in range(vectors.shape[1])])
        df_vectors['class_name'] = labels
        
        # Store in dictionary
        vector_storage[label] = df_vectors
        
        print(f"✅ TF-IDF shape {label}: {vectors.shape}")
    
    return vector_storage


from typing import Union, Tuple, List, Optional
import numpy as np

def prepare_tfidf_from_newsgroups(dataset: 'NewsgroupsDataset',
                                   split: str = 'train',
                                   use_preprocessed: bool = True,
                                   return_feature_names: bool = True,
                                   visualize: bool = False,
                                   method: str = 'pca',
                                   ngram_label: str = "(1,1)",
                                   selected_classes: Optional[List[str]] = None,
                                   save_dir: Optional[str] = None,
                                   **tfidf_params) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Generate TF-IDF vectors from a NewsgroupsDataset instance and optionally visualize them.

    Args:
        dataset (NewsgroupsDataset): The dataset instance.
        split (str): Dataset split ('train', 'validation', 'test', 'all').
        use_preprocessed (bool): Use tokenized text if True, raw text otherwise.
        return_feature_names (bool): Return feature names with TF-IDF matrix.
        visualize (bool): Whether to visualize the TF-IDF vectors using PCA, t-SNE, or 3D.
        method (str): 'pca', 'tsne', or '3d' for visualization method.
        ngram_label (str): A label for n-gram range used in visualization title.
        selected_classes (List[str], optional): If provided, limits visualization to specific classes.
        save_dir (str, optional): Directory to save visualization plots.
        **tfidf_params: Parameters passed to TfidfVectorizer.

    Returns:
        np.ndarray or (np.ndarray, List[str]): TF-IDF matrix and optionally feature names.
    """

    if use_preprocessed:
        df = dataset.create_dataframe(split=split, raw=False)
        texts = [' '.join(tokens) for tokens in df['data']]
    else:
        df = dataset.create_dataframe(split=split, raw=True)
        texts = df['text'].tolist()

    labels = df['class_name'].tolist()

    vectorizer = TFIDFVectorizer(preprocessed=False, **tfidf_params)
    vectors = vectorizer.fit_transform(texts)

    if visualize:
        n_components = 3 if method == '3d' else 2
        vectorizer.visualize_embeddings(
            vectors=vectors,
            labels=labels,
            method=method,
            n_components=n_components,
            title=f"{split.capitalize()} Split TF-IDF Embeddings",
            ngram_label=ngram_label,
            selected_classes=selected_classes,
            save_dir=save_dir
        )

    if return_feature_names:
        return vectors, vectorizer.get_feature_names()
    return vectors
