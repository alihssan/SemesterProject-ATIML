from typing import Union, Tuple, List, Optional
import numpy as np
from tf_idf import TFIDFVectorizer

def prepare_tfidf_from_newsgroups(dataset: 'NewsgroupsDataset',
                                   split: str = 'train',
                                   use_preprocessed: bool = True,
                                   return_feature_names: bool = True,
                                   visualize: bool = False,
                                   method: str = 'pca',
                                   ngram_label: str = "(1,1)",
                                   selected_classes: Optional[List[str]] = None,
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
            selected_classes=selected_classes
        )

    if return_feature_names:
        return vectors, vectorizer.get_feature_names()
    return vectors
