"""
Configuration for AG News Pipeline with Zero-Shot and Few-Shot Learning
"""

AG_NEWS_CONFIG = {
    'dataset': {
        'name': 'ag_news',
        'max_samples_per_class': 500,
        'test_size': 0.2,
        'val_size': 0.1,
        'few_shot_shots': [1, 3, 5, 10],
        'random_state': 42
    },
    
    'zero_shot': {
        'enabled': True,
        'model_name': 'facebook/bart-large-mnli',
        'prompts': [
            "This news article is about:",
            "The topic of this text is:",
            "This article discusses:",
            "The subject matter is:"
        ],
        'batch_size': 32,
        'device': 'auto'  # 'auto', 'cpu', or 'cuda'
    },
    
    'few_shot': {
        'enabled': True,
        'embedding_model': 'all-MiniLM-L6-v2',
        'classifier': 'logistic_regression',
        'classifier_params': {
            'random_state': 42,
            'max_iter': 1000,
            'C': 1.0
        }
    },
    
    'supervised': {
        'enabled': True,
        'vectorizer': 'tfidf',
        'vectorizer_params': {
            'max_features': 5000,
            'stop_words': 'english',
            'ngram_range': (1, 2)
        },
        'classifier': 'logistic_regression',
        'classifier_params': {
            'random_state': 42,
            'max_iter': 1000,
            'C': 1.0
        }
    },
    
    'evaluation': {
        'metrics': ['accuracy', 'f1_score', 'precision', 'recall'],
        'save_predictions': True,
        'save_confusion_matrix': True,
        'cross_validation_folds': 5
    },
    
    'visualization': {
        'create_comparison_chart': True,
        'create_learning_curves': True,
        'create_confusion_matrices': True,
        'figure_size': (12, 8),
        'dpi': 300
    },
    
    'logging': {
        'level': 'INFO',
        'save_logs': True,
        'log_file': 'ag_news_pipeline.log'
    }
}

# AG News class information
AG_NEWS_CLASSES = {
    0: 'World',
    1: 'Sports', 
    2: 'Business',
    3: 'Sci/Tech'
}

# Prompt templates for zero-shot classification
ZERO_SHOT_PROMPTS = {
    'simple': "This text is about:",
    'detailed': "The main topic of this news article is:",
    'question': "What is this article about?",
    'classification': "Classify this news article into one of these categories:"
}

# Few-shot learning configurations
FEW_SHOT_CONFIGS = {
    '1_shot': {
        'shots_per_class': 1,
        'description': 'One example per class'
    },
    '3_shot': {
        'shots_per_class': 3,
        'description': 'Three examples per class'
    },
    '5_shot': {
        'shots_per_class': 5,
        'description': 'Five examples per class'
    },
    '10_shot': {
        'shots_per_class': 10,
        'description': 'Ten examples per class'
    }
} 