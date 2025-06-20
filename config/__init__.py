"""
Configuration Module

This module provides centralized access to all hyperparameters and configuration
settings used throughout the ATIML project.
"""

from .config import (
    # Main configuration functions
    get_config_for_mode,
    update_config,
    
    # Individual configurations
    DATASET_CONFIG,
    TEXT_PREPROCESSING,
    TFIDF_CONFIG,
    DOC2VEC_CONFIG,
    LLM_CONFIG,
    SENTENCE_TRANSFORMER_CONFIG,
    CLASSIFIER_CONFIGS,
    SVM_CONFIG,
    NEURAL_NETWORK_CONFIG,
    RANDOM_FOREST_CONFIG,
    XGBOOST_CONFIG,
    VISUALIZATION_CONFIG,
    EVALUATION_CONFIG,
    PIPELINE_CONFIG,
    QUICK_MODE_CONFIG,
    ENVIRONMENT_CONFIG
)

__all__ = [
    'get_config_for_mode',
    'update_config',
    'DATASET_CONFIG',
    'TEXT_PREPROCESSING',
    'TFIDF_CONFIG',
    'DOC2VEC_CONFIG',
    'LLM_CONFIG',
    'SENTENCE_TRANSFORMER_CONFIG',
    'CLASSIFIER_CONFIGS',
    'SVM_CONFIG',
    'NEURAL_NETWORK_CONFIG',
    'RANDOM_FOREST_CONFIG',
    'XGBOOST_CONFIG',
    'VISUALIZATION_CONFIG',
    'EVALUATION_CONFIG',
    'PIPELINE_CONFIG',
    'QUICK_MODE_CONFIG',
    'ENVIRONMENT_CONFIG'
] 