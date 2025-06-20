"""
Hyperparameters Configuration

This module contains all hyperparameters used throughout the ATIML project.
Centralizing hyperparameters makes it easier to experiment and maintain consistency.
"""

# =============================================================================
# DATA LOADING AND PREPROCESSING PARAMETERS
# =============================================================================

# Dataset parameters
DATASET_CONFIG = {
    'categories': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'],
    'max_samples_per_category': 500,
    'test_size': 0.2,
    'random_state': 42,
    'remove': ('headers', 'footers', 'quotes')  # For 20 newsgroups
}

# Text preprocessing parameters
TEXT_PREPROCESSING = {
    'lowercase': True,
    'remove_special_chars': True,
    'remove_numbers': False,
    'remove_extra_whitespace': True,
    'min_text_length': 10
}

# =============================================================================
# TF-IDF VECTORIZATION PARAMETERS
# =============================================================================

TFIDF_CONFIG = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 2),
    'stop_words': 'english',
    'use_idf': True,
    'smooth_idf': True,
    'sublinear_tf': False,
    'token_pattern': r'(?u)\b\w\w+\b'  # Fixed: proper regex pattern as string
}

# =============================================================================
# DOC2VEC VECTORIZATION PARAMETERS
# =============================================================================

DOC2VEC_CONFIG = {
    'vector_size': 100,
    'window': 5,
    'min_count': 2,
    'epochs': 20,
    'workers': 4,
    'dm': 1,  # Distributed Memory
    'alpha': 0.025,
    'min_alpha': 0.0001,
    'negative': 5,
    'hs': 0,
    'sample': 1e-3
}

# =============================================================================
# LLM SUMMARIZATION PARAMETERS
# =============================================================================

LLM_CONFIG = {
    'model_name': "gemma:3b",
    'max_sentences': 4,
    'max_length': 150,
    'temperature': 0.7,
    'top_p': 0.9,
    'batch_size': 32
}

# =============================================================================
# SENTENCE TRANSFORMER PARAMETERS
# =============================================================================

SENTENCE_TRANSFORMER_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'batch_size': 32,
    'show_progress_bar': True,
    'normalize_embeddings': True
}

# =============================================================================
# CLASSIFIER PARAMETERS
# =============================================================================

# SVM Classifier
SVM_CONFIG = {
    'classifier_type': 'svm',
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'random_state': 42,
    'scale_features': True
}

# Neural Network Classifier
NEURAL_NETWORK_CONFIG = {
    'classifier_type': 'neural_network',
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'learning_rate': 'adaptive',
    'max_iter': 1000,
    'random_state': 42,
    'scale_features': True
}

# Random Forest Classifier
RANDOM_FOREST_CONFIG = {
    'classifier_type': 'random_forest',
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'scale_features': False
}

# XGBoost Classifier
XGBOOST_CONFIG = {
    'classifier_type': 'xgboost',
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'scale_features': False
}

# All classifier configurations
CLASSIFIER_CONFIGS = {
    'svm': SVM_CONFIG,
    'neural_network': NEURAL_NETWORK_CONFIG,
    'random_forest': RANDOM_FOREST_CONFIG,
    'xgboost': XGBOOST_CONFIG
}

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_format': 'png'
}

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================

EVALUATION_CONFIG = {
    'cross_validation_folds': 5,
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'random_state': 42
}

# =============================================================================
# PIPELINE PARAMETERS
# =============================================================================

PIPELINE_CONFIG = {
    'save_intermediate_results': True,
    'save_models': True,
    'save_visualizations': True,
    'verbose': True,
    'parallel_processing': False,
    'n_jobs': -1
}

# =============================================================================
# QUICK MODE PARAMETERS (for testing)
# =============================================================================

QUICK_MODE_CONFIG = {
    'max_samples_per_category': 50,
    'tfidf_max_features': 1000,
    'doc2vec_epochs': 5,
    'classifier_n_estimators': 50
}

# =============================================================================
# ENVIRONMENT PARAMETERS
# =============================================================================

ENVIRONMENT_CONFIG = {
    'results_dir': 'results',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'temp_dir': 'temp'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config_for_mode(mode='full'):
    """
    Get configuration based on mode (full, quick, etc.)
    
    Args:
        mode: Configuration mode ('full', 'quick')
        
    Returns:
        Dictionary with configuration for the specified mode
    """
    if mode == 'quick':
        # Override with quick mode settings
        config = {
            'dataset': {**DATASET_CONFIG, 'max_samples_per_category': QUICK_MODE_CONFIG['max_samples_per_category']},
            'tfidf': {**TFIDF_CONFIG, 'max_features': QUICK_MODE_CONFIG['tfidf_max_features']},
            'doc2vec': {**DOC2VEC_CONFIG, 'epochs': QUICK_MODE_CONFIG['doc2vec_epochs']},
            'classifiers': {name: {**config, 'n_estimators': QUICK_MODE_CONFIG['classifier_n_estimators']} 
                          for name, config in CLASSIFIER_CONFIGS.items()}
        }
    else:
        # Full configuration
        config = {
            'dataset': DATASET_CONFIG,
            'text_preprocessing': TEXT_PREPROCESSING,
            'tfidf': TFIDF_CONFIG,
            'doc2vec': DOC2VEC_CONFIG,
            'llm': LLM_CONFIG,
            'sentence_transformer': SENTENCE_TRANSFORMER_CONFIG,
            'classifiers': CLASSIFIER_CONFIGS,
            'visualization': VISUALIZATION_CONFIG,
            'evaluation': EVALUATION_CONFIG,
            'pipeline': PIPELINE_CONFIG,
            'environment': ENVIRONMENT_CONFIG
        }
    
    return config

def update_config(base_config, updates):
    """
    Update a configuration dictionary with new values.
    
    Args:
        base_config: Base configuration dictionary
        updates: Dictionary with updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    import copy
    updated_config = copy.deepcopy(base_config)
    
    for key, value in updates.items():
        if key in updated_config and isinstance(updated_config[key], dict) and isinstance(value, dict):
            updated_config[key].update(value)
        else:
            updated_config[key] = value
    
    return updated_config
