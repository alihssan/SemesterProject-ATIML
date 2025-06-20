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


class Doc2VecVectorizer:
    """
    A comprehensive Doc2Vec vectorizer class for document vectorization.
    """
    
    def __init__(self,
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 1,
                 dm: int = 1,
                 dbow_words: int = 0,
                 dm_mean: int = 0,
                 dm_concat: int = 0,
                 dm_tag_count: int = 1,
                 alpha: float = 0.025,
                 min_alpha: float = 0.0001,
                 seed: int = 1,
                 workers: int = 3,
                 epochs: int = 20,
                 hs: int = 0,
                 negative: int = 5,
                 ns_exponent: float = 0.75,
                 cbow_mean: int = 1,
                 compute_loss: bool = False,
                 callbacks: Optional[List] = None,
                 max_vocab_size: Optional[int] = None,
                 max_final_vocab: Optional[int] = None,
                 sample: float = 1e-3,
                 hashfxn: Optional[callable] = None,
                 trim_rule: Optional[callable] = None,
                 sorted_vocab: int = 1,
                 batch_words: int = 10000,
                 **kwargs):
        """
        Initialize Doc2Vec Vectorizer with specified parameters.
        
        Args:
            vector_size: Dimensionality of the feature vectors
            window: Maximum distance between the current and predicted word
            min_count: Ignores all words with total frequency lower than this
            dm: Defines the training algorithm (1 for PV-DM, 0 for PV-DBOW)
            dbow_words: If set to 1, trains word-vectors using skip-gram
            dm_mean: If 0, use the sum of the context word vectors
            dm_concat: If 1, use concatenation of context vectors
            dm_tag_count: Expected constant number of document tags per document
            alpha: Initial learning rate
            min_alpha: Final learning rate
            seed: Seed for random number generator
            workers: Number of worker threads
            epochs: Number of iterations over the corpus
            hs: If 1, hierarchical softmax will be used
            negative: If > 0, negative sampling will be used
            ns_exponent: Exponent used to shape the negative sampling distribution
            cbow_mean: If 0, use the sum of the context word vectors
            compute_loss: If True, computes and stores loss value
            callbacks: List of callbacks to be called during training
            max_vocab_size: Maximum vocabulary size
            max_final_vocab: Maximum final vocabulary size
            sample: Threshold for configuring which higher-frequency words are randomly downsampled
            hashfxn: Hash function to use for randomizing
            trim_rule: Vocabulary trimming rule
            sorted_vocab: If 1, sort the vocabulary by descending frequency
            batch_words: Target size for batches of words
            **kwargs: Additional arguments passed to Doc2Vec
        """
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
        
        # Set up logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for Doc2Vec training.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        return simple_preprocess(text, deacc=True)
    
    def _create_tagged_documents(self, texts: Union[List[str], pd.Series, np.ndarray], 
                                tags: Optional[List[str]] = None) -> List[TaggedDocument]:
        """
        Create TaggedDocument objects for Doc2Vec training.
        
        Args:
            texts: List of text documents
            tags: Optional list of tags for documents
            
        Returns:
            List of TaggedDocument objects
        """
        tagged_docs = []
        
        for i, text in enumerate(texts):
            if isinstance(text, (np.ndarray, pd.Series)):
                text = str(text)
            
            # Preprocess text
            tokens = self._preprocess_text(text)
            
            # Create tag
            if tags is not None and i < len(tags):
                tag = tags[i]
            else:
                tag = f"DOC_{i}"
            
            # Create TaggedDocument
            tagged_doc = TaggedDocument(words=tokens, tags=[tag])
            tagged_docs.append(tagged_doc)
        
        return tagged_docs
    
    def fit(self, texts: Union[List[str], pd.Series, np.ndarray],
            tags: Optional[List[str]] = None,
            **kwargs) -> 'Doc2VecVectorizer':
        """
        Fit the Doc2Vec model on the given texts.
        
        Args:
            texts: List of text documents to train on
            tags: Optional list of tags for documents
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        # Create tagged documents
        tagged_docs = self._create_tagged_documents(texts, tags)
        
        # Initialize model
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            dm=self.dm,
            dbow_words=self.dbow_words,
            dm_mean=self.dm_mean,
            dm_concat=self.dm_concat,
            dm_tag_count=self.dm_tag_count,
            alpha=self.alpha,
            min_alpha=self.min_alpha,
            seed=self.seed,
            workers=self.workers,
            hs=self.hs,
            negative=self.negative,
            ns_exponent=self.ns_exponent,
            cbow_mean=self.cbow_mean,
            compute_loss=self.compute_loss,
            callbacks=self.callbacks,
            max_vocab_size=self.max_vocab_size,
            max_final_vocab=self.max_final_vocab,
            sample=self.sample,
            hashfxn=self.hashfxn,
            trim_rule=self.trim_rule,
            sorted_vocab=self.sorted_vocab,
            batch_words=self.batch_words,
            **self.kwargs
        )
        
        # Build vocabulary
        self.model.build_vocab(tagged_docs)
        self.vocabulary_size = len(self.model.wv.key_to_index)
        
        # Train model
        epochs = kwargs.get('epochs', self.epochs)
        self.model.train(
            tagged_docs,
            total_examples=self.model.corpus_count,
            epochs=epochs
        )
        
        self.is_trained = True
        self.document_count = len(tagged_docs)
        
        return self
    
    def transform(self, texts: Union[List[str], pd.Series, np.ndarray],
                  tags: Optional[List[str]] = None,
                  infer_steps: int = 20,
                  alpha: Optional[float] = None,
                  min_alpha: Optional[float] = None) -> np.ndarray:
        """
        Transform texts to Doc2Vec vectors.
        
        Args:
            texts: List of text documents to transform
            tags: Optional list of tags for documents
            infer_steps: Number of inference steps
            alpha: Learning rate for inference
            min_alpha: Minimum learning rate for inference
            
        Returns:
            Doc2Vec vectors as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before transforming data")
        
        vectors = []
        
        for i, text in enumerate(texts):
            if isinstance(text, (np.ndarray, pd.Series)):
                text = str(text)
            
            # Preprocess text
            tokens = self._preprocess_text(text)
            
            # Infer vector with minimal parameters to avoid compatibility issues
            try:
                # Try with epochs parameter (newer gensim versions)
                vector = self.model.infer_vector(
                    tokens,
                    epochs=infer_steps,
                    alpha=alpha or self.alpha,
                    min_alpha=min_alpha or self.min_alpha
                )
            except TypeError:
                # Fallback to basic inference without additional parameters
                vector = self.model.infer_vector(tokens)
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: Union[List[str], pd.Series, np.ndarray],
                     tags: Optional[List[str]] = None,
                     **kwargs) -> np.ndarray:
        """
        Fit the model and transform the texts in one step.
        
        Args:
            texts: List of text documents to fit and transform
            tags: Optional list of tags for documents
            **kwargs: Additional parameters
            
        Returns:
            Doc2Vec vectors as numpy array
        """
        self.fit(texts, tags, **kwargs)
        return self.transform(texts, tags)
    
    def get_document_vector(self, doc_id: str) -> np.ndarray:
        """
        Get the vector for a specific document by its tag.
        
        Args:
            doc_id: Document tag/ID
            
        Returns:
            Document vector
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting document vectors")
        
        return self.model.dv[doc_id]
    
    def get_similar_documents(self, doc_id: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find the most similar documents to a given document.
        
        Args:
            doc_id: Document tag/ID
            topn: Number of similar documents to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar documents")
        
        return self.model.dv.most_similar(doc_id, topn=topn)
    
    def get_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find the most similar words to a given word.
        
        Args:
            word: Input word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar words")
        
        return self.model.wv.most_similar(word, topn=topn)
    
    def get_vocabulary(self) -> List[str]:
        """
        Get the vocabulary (list of words) from the trained model.
        
        Returns:
            List of vocabulary words
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting vocabulary")
        
        return list(self.model.wv.key_to_index.keys())
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of words in vocabulary
        """
        return self.vocabulary_size
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> 'Doc2VecVectorizer':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = Doc2Vec.load(filepath)
        self.is_trained = True
        self.vocabulary_size = len(self.model.wv.key_to_index)
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
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


def create_doc2vec_vectors(texts: Union[List[str], pd.Series, np.ndarray],
                          vector_size: int = 100,
                          window: int = 5,
                          min_count: int = 1,
                          dm: int = 1,
                          dbow_words: int = 0,
                          dm_mean: int = 0,
                          dm_concat: int = 0,
                          dm_tag_count: int = 1,
                          alpha: float = 0.025,
                          min_alpha: float = 0.0001,
                          seed: int = 1,
                          workers: int = 3,
                          epochs: int = 20,
                          hs: int = 0,
                          negative: int = 5,
                          ns_exponent: float = 0.75,
                          cbow_mean: int = 1,
                          compute_loss: bool = False,
                          callbacks: Optional[List] = None,
                          max_vocab_size: Optional[int] = None,
                          max_final_vocab: Optional[int] = None,
                          sample: float = 1e-3,
                          hashfxn: Optional[callable] = None,
                          trim_rule: Optional[callable] = None,
                          sorted_vocab: int = 1,
                          batch_words: int = 10000,
                          tags: Optional[List[str]] = None,
                          return_model: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Doc2VecVectorizer]]:
    """
    Create Doc2Vec vectors from text data.
    
    Args:
        texts: List of text documents to vectorize
        vector_size: Dimensionality of the feature vectors
        window: Maximum distance between the current and predicted word
        min_count: Ignores all words with total frequency lower than this
        dm: Defines the training algorithm (1 for PV-DM, 0 for PV-DBOW)
        dbow_words: If set to 1, trains word-vectors using skip-gram
        dm_mean: If 0, use the sum of the context word vectors
        dm_concat: If 1, use concatenation of context vectors
        dm_tag_count: Expected constant number of document tags per document
        alpha: Initial learning rate
        min_alpha: Final learning rate
        seed: Seed for random number generator
        workers: Number of worker threads
        epochs: Number of iterations over the corpus
        hs: If 1, hierarchical softmax will be used
        negative: If > 0, negative sampling will be used
        ns_exponent: Exponent used to shape the negative sampling distribution
        cbow_mean: If 0, use the sum of the context word vectors
        compute_loss: If True, computes and stores loss value
        callbacks: List of callbacks to be called during training
        max_vocab_size: Maximum vocabulary size
        max_final_vocab: Maximum final vocabulary size
        sample: Threshold for configuring which higher-frequency words are randomly downsampled
        hashfxn: Hash function to use for randomizing
        trim_rule: Vocabulary trimming rule
        sorted_vocab: If 1, sort the vocabulary by descending frequency
        batch_words: Target size for batches of words
        tags: Optional list of tags for documents
        return_model: Whether to return the trained model along with vectors
        
    Returns:
        Doc2Vec vectors as numpy array, optionally with the trained model
    """
    vectorizer = Doc2VecVectorizer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        dm=dm,
        dbow_words=dbow_words,
        dm_mean=dm_mean,
        dm_concat=dm_concat,
        dm_tag_count=dm_tag_count,
        alpha=alpha,
        min_alpha=min_alpha,
        seed=seed,
        workers=workers,
        epochs=epochs,
        hs=hs,
        negative=negative,
        ns_exponent=ns_exponent,
        cbow_mean=cbow_mean,
        compute_loss=compute_loss,
        callbacks=callbacks,
        max_vocab_size=max_vocab_size,
        max_final_vocab=max_final_vocab,
        sample=sample,
        hashfxn=hashfxn,
        trim_rule=trim_rule,
        sorted_vocab=sorted_vocab,
        batch_words=batch_words
    )
    
    vectors = vectorizer.fit_transform(texts, tags)
    
    if return_model:
        return vectors, vectorizer
    else:
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
