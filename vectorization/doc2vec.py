"""
Doc2Vec Vectorization Module

This module provides a class to create Doc2Vec vectors from text data.
Doc2Vec (Document to Vector) is an unsupervised algorithm that learns
fixed-length feature representations from variable-length pieces of texts.
"""

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from typing import Union, List, Optional, Tuple, Dict, Any
import pickle
import os
import logging
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


class Doc2VecVectorizer:
    def __init__(self, vector_size=100, window=5, min_count=1, dm=1, dbow_words=0, dm_mean=0, dm_concat=0,
                 dm_tag_count=1, alpha=0.025, min_alpha=0.0001, seed=1, workers=3, epochs=20, hs=0,
                 negative=5, ns_exponent=0.75, cbow_mean=1, compute_loss=False, callbacks=None,
                 max_vocab_size=None, max_final_vocab=None, sample=1e-3, hashfxn=None, trim_rule=None,
                 sorted_vocab=1, batch_words=10000, **kwargs):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_mean = dm_mean
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.seed = seed
        self.workers = workers
        self.epochs = epochs
        self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.max_vocab_size = max_vocab_size
        self.max_final_vocab = max_final_vocab
        self.sample = sample
        self.hashfxn = hashfxn
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.kwargs = kwargs

        self.model = None
        self.is_trained = False
        self.vocabulary_size = 0
        self.document_count = 0

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def _preprocess_text(self, text):
        return simple_preprocess(text, deacc=True)

    def _create_tagged_documents(self, texts, tags=None):
        tagged_docs = []
        for i, text in enumerate(texts):
            if isinstance(text, (np.ndarray, pd.Series)):
                text = str(text)
            tokens = self._preprocess_text(text)
            tag = tags[i] if tags and i < len(tags) else f"DOC_{i}"
            tagged_docs.append(TaggedDocument(words=tokens, tags=[tag]))
        return tagged_docs

    def fit(self, texts, tags=None, **kwargs):
        tagged_docs = self._create_tagged_documents(texts, tags)
        self.model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count,
                             dm=self.dm, dbow_words=self.dbow_words, dm_mean=self.dm_mean,
                             dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count, alpha=self.alpha,
                             min_alpha=self.min_alpha, seed=self.seed, workers=self.workers, hs=self.hs,
                             negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
                             compute_loss=self.compute_loss, callbacks=self.callbacks,
                             max_vocab_size=self.max_vocab_size, max_final_vocab=self.max_final_vocab,
                             sample=self.sample, hashfxn=self.hashfxn, trim_rule=self.trim_rule,
                             sorted_vocab=self.sorted_vocab, batch_words=self.batch_words, **self.kwargs)

        self.model.build_vocab(tagged_docs)
        self.vocabulary_size = len(self.model.wv.key_to_index)
        self.model.train(tagged_docs, total_examples=self.model.corpus_count, epochs=kwargs.get('epochs', self.epochs))
        self.is_trained = True
        self.document_count = len(tagged_docs)
        return self

    def transform(self, texts, tags=None, infer_steps=20, alpha=None, min_alpha=None):
        if not self.is_trained:
            raise ValueError("Model must be trained before transforming data")
        vectors = []
        for i, text in enumerate(texts):
            if isinstance(text, (np.ndarray, pd.Series)):
                text = str(text)
            tokens = self._preprocess_text(text)
            try:
                vector = self.model.infer_vector(tokens, epochs=infer_steps,
                                                 alpha=alpha or self.alpha,
                                                 min_alpha=min_alpha or self.min_alpha)
            except TypeError:
                vector = self.model.infer_vector(tokens)
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, texts, tags=None, **kwargs):
        self.fit(texts, tags, **kwargs)
        return self.transform(texts, tags)

    def visualize_embeddings(self, vectors: np.ndarray, labels: List[str], method: str = 'pca',
                             n_components: int = 2, title: str = "Doc2Vec Embeddings",
                             selected_classes: Optional[List[str]] = None) -> None:
        df = pd.DataFrame(vectors)
        df['label'] = labels
        if selected_classes:
            df = df[df['label'].isin(selected_classes)]
        labels_filtered = df['label'].tolist()
        X_filtered = normalize(df.drop(columns='label').values)
        if method == '3d':
            if n_components != 3:
                raise ValueError("For 3D visualization, n_components must be 3")
            reducer = PCA(n_components=3)
            X_reduced = reducer.fit_transform(X_filtered)
            fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                color=labels_filtered,
                                title=f"3D PCA Doc2Vec Embeddings | {title}",
                                labels={'color': 'Class'})
            fig.show()
            return
        if method == 'tsne':
            X_pca = PCA(n_components=min(50, X_filtered.shape[1])).fit_transform(X_filtered)
            reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
            X_reduced = reducer.fit_transform(X_pca)
        else:
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X_filtered)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels_filtered,
                        palette='tab20', legend='full', s=25)
        plt.title(f"{method.upper()} Doc2Vec Visualization\n{title}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=2, fontsize='small')
        plt.tight_layout()
        plt.show()

    def get_document_vector(self, doc_id: str):
        if not self.is_trained:
            raise ValueError("Model must be trained before getting document vectors")
        return self.model.dv[doc_id]

    def get_similar_documents(self, doc_id: str, topn: int = 10):
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar documents")
        return self.model.dv.most_similar(doc_id, topn=topn)

    def get_similar_words(self, word: str, topn: int = 10):
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar words")
        return self.model.wv.most_similar(word, topn=topn)

    def get_vocabulary(self):
        if not self.is_trained:
            raise ValueError("Model must be trained before getting vocabulary")
        return list(self.model.wv.key_to_index.keys())

    def get_vocabulary_size(self):
        return self.vocabulary_size

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        self.model.save(filepath)

    def load_model(self, filepath: str) -> 'Doc2VecVectorizer':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = Doc2Vec.load(filepath)
        self.is_trained = True
        self.vocabulary_size = len(self.model.wv.key_to_index)
        return self

    def get_model_info(self):
        if not self.is_trained:
            return {"status": "Model not trained"}
        return {
            "vector_size": self.model.vector_size,
            "vocabulary_size": self.vocabulary_size,
            "document_count": self.document_count,
            "window": self.model.window,
            "min_count": self.model.min_count,
            "dm": self.model.dm,
            "alpha": self.model.alpha,
            "min_alpha": self.model.min_alpha,
            "epochs": self.epochs,
            "is_trained": self.is_trained
        }

def create_doc2vec_vectors_from_newsgroups(dataset: 'NewsgroupsDataset',
                                            split: str = 'train',
                                            use_preprocessed: bool = True,
                                            visualize: bool = False,
                                            method: str = 'pca',
                                            selected_classes: Optional[List[str]] = None,
                                            vector_size: int = 100,
                                            epochs: int = 20,
                                            return_model: bool = False,
                                            **doc2vec_params) -> Union[np.ndarray, Tuple[np.ndarray, Doc2VecVectorizer]]:
    """
    Create Doc2Vec vectors from a NewsgroupsDataset instance and optionally visualize them.

    Args:
        dataset: Instance of NewsgroupsDataset.
        split: 'train', 'validation', 'test', or 'all'.
        use_preprocessed: Use tokenized text or raw.
        visualize: Whether to visualize the embeddings.
        method: 'pca', 'tsne', or '3d'.
        selected_classes: Subset of classes to visualize.
        vector_size: Dimension of the vectors.
        epochs: Training epochs.
        return_model: If True, return model with vectors.
        **doc2vec_params: Additional parameters.

    Returns:
        Doc2Vec vectors or (vectors, model).
    """
    if use_preprocessed:
        df = dataset.create_dataframe(split=split, raw=False)
        texts = [' '.join(tokens) for tokens in df['data']]
    else:
        df = dataset.create_dataframe(split=split, raw=True)
        texts = df['text'].tolist()

    labels = df['class_name'].tolist()

    model = Doc2VecVectorizer(vector_size=vector_size, epochs=epochs, **doc2vec_params)
    vectors = model.fit_transform(texts)

    if visualize:
        n_components = 3 if method == '3d' else 2
        model.visualize_embeddings(
            vectors=vectors,
            labels=labels,
            method=method,
            n_components=n_components,
            title=f"{split.capitalize()} Doc2Vec Embeddings",
            selected_classes=selected_classes
        )

    if return_model:
        return vectors, model
    return vectors


def analyze_doc2vec_vectors(vectors: np.ndarray,
                           tags: Optional[List[str]] = None,
                           top_k: int = 10) -> pd.DataFrame:
    """
    Analyze Doc2Vec vectors and return statistics.

    Args:
        vectors: Doc2Vec vectors
        tags: Optional list of document tags
        top_k: Number of top documents to analyze

    Returns:
        DataFrame with vector analysis
    """
    # Calculate statistics
    mean_vector = np.mean(vectors, axis=0)
    std_vector = np.std(vectors, axis=0)
    max_vector = np.max(vectors, axis=0)
    min_vector = np.min(vectors, axis=0)

    # Calculate vector magnitudes
    magnitudes = np.linalg.norm(vectors, axis=1)

    # Create analysis DataFrame
    analysis_data = {
        'magnitude': magnitudes,
        'mean_component': np.mean(vectors, axis=1),
        'std_component': np.std(vectors, axis=1),
        'max_component': np.max(vectors, axis=1),
        'min_component': np.min(vectors, axis=1),
        'component_range': np.max(vectors, axis=1) - np.min(vectors, axis=1)
    }

    if tags:
        analysis_data['tag'] = tags

    analysis_df = pd.DataFrame(analysis_data)

    # Sort by magnitude and return top k
    return analysis_df.sort_values('magnitude', ascending=False).head(top_k)


if __name__ == "__main__":
    pass
