"""
Main Pipeline for 20 Newsgroups Analysis

This pipeline implements the complete workflow:
1. Load and Preprocess 20 Newsgroups
2. Generate 3 Document Representations (TF-IDF, Doc2Vec, Gemini + Sentence-BERT)
3. Train Classifiers (SVM, MLP, Decision Trees)
4. Generate Gemini reasoning for predictions
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Import configuration
from config import get_config_for_mode, DATASET_CONFIG, TFIDF_CONFIG, DOC2VEC_CONFIG, LLM_CONFIG, SENTENCE_TRANSFORMER_CONFIG, CLASSIFIER_CONFIGS, VISUALIZATION_CONFIG

# Import our custom modules
from vectorization.tf_idf import create_tfidf_vectors
from vectorization.doc2vec import create_doc2vec_vectors
from vectorization.llm_summarisation import create_summaries
from train.classifier import UnifiedClassifier, train_test_classifier

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: Sentence Transformers not available. Using basic Doc2Vec only.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsgroupsPipeline:
    """
    Complete pipeline for 20 Newsgroups analysis.
    """
    
    def __init__(self, 
                 categories: List[str] = None,
                 max_samples_per_category: int = None,
                 test_size: float = None,
                 random_state: int = None,
                 config_mode: str = 'full',
                 custom_config: Dict = None):
        """
        Initialize the pipeline.
        
        Args:
            categories: Newsgroup categories to use (None for all)
            max_samples_per_category: Maximum samples per category
            test_size: Proportion for test set
            random_state: Random seed
            config_mode: Configuration mode ('full', 'quick')
            custom_config: Custom configuration dictionary to override defaults
        """
        # Load configuration
        self.config = get_config_for_mode(config_mode)
        
        # Override with custom config if provided
        if custom_config:
            from config import update_config
            self.config = update_config(self.config, custom_config)
        
        # Set parameters (use config defaults if not provided)
        self.categories = categories or self.config['dataset']['categories']
        self.max_samples_per_category = max_samples_per_category or self.config['dataset']['max_samples_per_category']
        self.test_size = test_size or self.config['dataset']['test_size']
        self.random_state = random_state or self.config['dataset']['random_state']
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Vector representations
        self.tfidf_vectors = None
        self.doc2vec_vectors = None
        self.gemini_summaries = None
        self.sentence_bert_vectors = None
        
        # Classifiers
        self.classifiers = {}
        self.classifier_results = {}
        
        # XAI components
        self.sentence_transformer = None
        
        # Create results directory
        self.results_dir = f"results/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Pipeline initialized. Results will be saved to: {self.results_dir}")
        logger.info(f"Configuration mode: {config_mode}")
        logger.info(f"Categories: {self.categories}")
        logger.info(f"Max samples per category: {self.max_samples_per_category}")
    
    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess the 20 Newsgroups dataset.
        """
        logger.info("Loading 20 Newsgroups dataset...")
        
        # Load dataset
        newsgroups = fetch_20newsgroups(
            subset='all',
            categories=self.categories,
            remove=self.config['dataset']['remove'],
            random_state=self.random_state
        )
        
        # Create DataFrame
        self.raw_data = pd.DataFrame({
            'text': newsgroups.data,
            'target': newsgroups.target,
            'target_name': [newsgroups.target_names[i] for i in newsgroups.target]
        })
        
        # Limit samples per category if specified
        if self.max_samples_per_category:
            self.raw_data = self.raw_data.groupby('target').head(self.max_samples_per_category).reset_index(drop=True)
        
        # Basic text preprocessing
        self.processed_data = self.raw_data.copy()
        self.processed_data['text_clean'] = self.processed_data['text'].apply(self._clean_text)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.processed_data['text_clean'],
            self.processed_data['target'],
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.processed_data['target']
        )
        
        logger.info(f"Data loaded: {len(self.raw_data)} samples, {len(self.raw_data['target'].unique())} categories")
        logger.info(f"Train set: {len(self.X_train)}, Test set: {len(self.X_test)}")
        
        # Save data info
        self._save_data_info()
    
    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        import re
        
        # Get text preprocessing config
        text_config = self.config.get('text_preprocessing', {})
        
        # Convert to lowercase if specified
        if text_config.get('lowercase', True):
            text = text.lower()
        
        # Remove special characters but keep spaces
        if text_config.get('remove_special_chars', True):
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove numbers if specified
        if text_config.get('remove_numbers', False):
            text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        if text_config.get('remove_extra_whitespace', True):
            text = ' '.join(text.split())
        
        return text
    
    def _save_data_info(self) -> None:
        """Save information about the dataset."""
        info = {
            'total_samples': len(self.raw_data),
            'categories': self.raw_data['target_name'].unique().tolist(),
            'category_counts': self.raw_data['target_name'].value_counts().to_dict(),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        with open(f"{self.results_dir}/data_info.json", 'w') as f:
            import json
            json.dump(info, f, indent=2)
    
    def generate_tfidf_vectors(self) -> None:
        """
        Generate TF-IDF vectors.
        """
        logger.info("Generating TF-IDF vectors...")
        
        # Combine train and test for fitting
        all_texts = pd.concat([self.X_train, self.X_test])
        
        # Get TF-IDF configuration
        tfidf_config = self.config['tfidf']
        
        # Generate TF-IDF vectors
        self.tfidf_vectors, feature_names = create_tfidf_vectors(
            all_texts,
            max_features=tfidf_config['max_features'],
            min_df=tfidf_config['min_df'],
            max_df=tfidf_config['max_df'],
            ngram_range=tfidf_config['ngram_range'],
            stop_words=tfidf_config.get('stop_words', 'english'),
            use_idf=tfidf_config.get('use_idf', True),
            smooth_idf=tfidf_config.get('smooth_idf', True),
            sublinear_tf=tfidf_config.get('sublinear_tf', False),
            token_pattern=tfidf_config.get('token_pattern', r'(?u)\b\w\w+\b'),
            return_feature_names=True
        )
        
        # Split back into train and test
        train_size = len(self.X_train)
        self.tfidf_train = self.tfidf_vectors[:train_size]
        self.tfidf_test = self.tfidf_vectors[train_size:]
        
        logger.info(f"TF-IDF vectors generated: {self.tfidf_vectors.shape}")
        
        # Save feature names
        with open(f"{self.results_dir}/tfidf_features.pkl", 'wb') as f:
            pickle.dump(feature_names, f)
    
    def generate_doc2vec_vectors(self) -> None:
        """
        Generate Doc2Vec vectors.
        """
        logger.info("Generating Doc2Vec vectors...")
        
        # Combine train and test for training
        all_texts = pd.concat([self.X_train, self.X_test])
        
        # Get Doc2Vec configuration
        doc2vec_config = self.config['doc2vec']
        
        # Generate Doc2Vec vectors
        self.doc2vec_vectors = create_doc2vec_vectors(
            all_texts,
            vector_size=doc2vec_config['vector_size'],
            window=doc2vec_config['window'],
            min_count=doc2vec_config['min_count'],
            epochs=doc2vec_config['epochs'],
            workers=doc2vec_config['workers'],
            dm=doc2vec_config.get('dm', 1),
            alpha=doc2vec_config.get('alpha', 0.025),
            min_alpha=doc2vec_config.get('min_alpha', 0.0001),
            negative=doc2vec_config.get('negative', 5),
            hs=doc2vec_config.get('hs', 0),
            sample=doc2vec_config.get('sample', 1e-3)
        )
        
        # Split back into train and test
        train_size = len(self.X_train)
        self.doc2vec_train = self.doc2vec_vectors[:train_size]
        self.doc2vec_test = self.doc2vec_vectors[train_size:]
        
        logger.info(f"Doc2Vec vectors generated: {self.doc2vec_vectors.shape}")
    
    def generate_gemini_summaries_and_bert_vectors(self) -> None:
        """
        Generate Gemini summaries and convert to Sentence-BERT vectors.
        """
        logger.info("Generating Gemini summaries...")
        
        # Get LLM configuration
        llm_config = self.config['llm']
        
        # Generate summaries for all texts
        all_texts = pd.concat([self.X_train, self.X_test])
        self.gemini_summaries = create_summaries(
            all_texts,
            model_name=llm_config['model_name'],
            max_sentences=llm_config['max_sentences'],
            max_length=llm_config.get('max_length', 150),
            temperature=llm_config.get('temperature', 0.7),
            top_p=llm_config.get('top_p', 0.9),
            batch_size=llm_config.get('batch_size', 32)
        )
        
        # Convert summaries to vectors
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("Converting summaries to Sentence-BERT vectors...")
            
            # Get Sentence Transformer configuration
            st_config = self.config['sentence_transformer']
            
            # Initialize Sentence-BERT
            self.sentence_transformer = SentenceTransformer(st_config['model_name'])
            
            # Convert summaries to vectors
            self.sentence_bert_vectors = self.sentence_transformer.encode(
                self.gemini_summaries,
                show_progress_bar=st_config.get('show_progress_bar', True),
                batch_size=st_config.get('batch_size', 32),
                normalize_embeddings=st_config.get('normalize_embeddings', True)
            )
            
            # Split back into train and test
            train_size = len(self.X_train)
            self.bert_train = self.sentence_bert_vectors[:train_size]
            self.bert_test = self.sentence_bert_vectors[train_size:]
            
            logger.info(f"Sentence-BERT vectors generated: {self.sentence_bert_vectors.shape}")
        else:
            logger.warning("Sentence Transformers not available - skipping Sentence-BERT vectors")
            self.sentence_bert_vectors = None
            self.bert_train = None
            self.bert_test = None
        
        # Save summaries
        summary_df = pd.DataFrame({
            'original_text': all_texts,
            'summary': self.gemini_summaries
        })
        summary_df.to_csv(f"{self.results_dir}/summaries.csv", index=False)
    
    def train_classifiers(self) -> None:
        """
        Train classifiers on all three representations.
        """
        logger.info("Training classifiers...")
        
        # Define available representations
        representations = {
            'tfidf': (self.tfidf_train, self.tfidf_test),
            'doc2vec': (self.doc2vec_train, self.doc2vec_test),
        }
        
        # Add sentence-BERT if available
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.bert_train is not None:
            representations['sentence_bert'] = (self.bert_train, self.bert_test)
        else:
            logger.info("Skipping sentence-BERT training (not available)")
        
        # Get classifier configurations
        classifier_configs = self.config['classifiers']
        
        for rep_name, (X_train_rep, X_test_rep) in representations.items():
            logger.info(f"Training classifiers on {rep_name} representation...")
            
            self.classifiers[rep_name] = {}
            self.classifier_results[rep_name] = {}
            
            for clf_name, config in classifier_configs.items():
                logger.info(f"Training {clf_name} on {rep_name}...")
                
                # Train classifier
                classifier, results = train_test_classifier(
                    X_train_rep, self.y_train,
                    classifier_type=config['classifier_type'],
                    test_size=0.2,
                    scale_features=config.get('scale_features', True),
                    **{k: v for k, v in config.items() if k not in ['classifier_type', 'scale_features']}
                )
                
                # Evaluate on test set
                test_results = classifier.evaluate(X_test_rep, self.y_test)
                
                # Store results
                self.classifiers[rep_name][clf_name] = classifier
                self.classifier_results[rep_name][clf_name] = {
                    'train_results': results,
                    'test_results': test_results
                }
                
                logger.info(f"{clf_name} on {rep_name} - Test Accuracy: {test_results['accuracy']:.4f}")
        
        # Save results
        self._save_classifier_results()
    
    def _save_classifier_results(self) -> None:
        """Save classifier results to files."""
        results_summary = {}
        
        for rep_name, classifiers in self.classifier_results.items():
            results_summary[rep_name] = {}
            for clf_name, results in classifiers.items():
                results_summary[rep_name][clf_name] = {
                    'test_accuracy': results['test_results']['accuracy'],
                    'train_accuracy': results['train_results']['accuracy']
                }
        
        # Save summary
        with open(f"{self.results_dir}/classifier_results.json", 'w') as f:
            import json
            json.dump(results_summary, f, indent=2)
        
        # Save detailed results
        with open(f"{self.results_dir}/detailed_results.pkl", 'wb') as f:
            pickle.dump(self.classifier_results, f)
    
    def generate_gemini_reasoning(self) -> Dict[str, Any]:
        """
        Generate Gemini reasoning for predictions.
            
        Returns:
            Dictionary with Gemini reasoning
        """
        logger.info("Generating Gemini reasoning for predictions...")
        
        # Select random samples for analysis
            sample_indices = np.random.choice(len(self.X_test), min(5, len(self.X_test)), replace=False)
        
        gemini_analysis = {}
        
        for i, idx in enumerate(sample_indices):
            logger.info(f"Analyzing sample {i+1}/{len(sample_indices)}...")
            
            # Get sample data
            text = self.X_test.iloc[idx]
            true_label = self.y_test.iloc[idx]
            true_label_name = self.raw_data['target_name'].iloc[true_label]
            
            # Get prediction from best classifier (using TF-IDF Random Forest)
            tfidf_vector = self.tfidf_test[idx].reshape(1, -1)
            rf_classifier = self.classifiers['tfidf']['random_forest']
            prediction = rf_classifier.predict(tfidf_vector)[0]
            prediction_name = self.raw_data['target_name'].iloc[prediction]
            
            # Get prediction confidence
                try:
                    proba = rf_classifier.predict_proba(tfidf_vector)[0]
                    confidence = proba[prediction]
            except:
                confidence = 0.0
            
            # Create prompt for Gemini
            prompt = f"""
            Analyze this text classification prediction and provide reasoning:
            
            Text: {text}
            True Label: {true_label_name}
            Predicted Label: {prediction_name}
            Prediction Confidence: {confidence:.3f}
            
            Please provide:
            1. Your reasoning for why this text belongs to the predicted category
            2. Whether you agree with the model's prediction
            3. Any additional insights about the classification
            """
            
            # Generate Gemini reasoning
            try:
                from vectorization.llm_summarisation import OllamaSummarizer
                summarizer = OllamaSummarizer()
                reasoning = summarizer.summarize(prompt, max_sentences=6)
            except Exception as e:
                reasoning = f"Error generating reasoning: {str(e)}"
            
            gemini_analysis[f'sample_{i}'] = {
                'text': text,
                'true_label': true_label_name,
                'predicted_label': prediction_name,
                'confidence': confidence,
                'correct': true_label == prediction,
                'gemini_reasoning': reasoning
            }
        
        # Save Gemini analysis
        with open(f"{self.results_dir}/gemini_analysis.pkl", 'wb') as f:
            pickle.dump(gemini_analysis, f)
        
        logger.info("Gemini reasoning completed and saved")
        return gemini_analysis
    
    def create_visualizations(self) -> None:
        """
        Create visualizations for the pipeline results.
        """
        logger.info("Creating visualizations...")
        
        # 1. Comprehensive metrics chart (all 4 metrics)
        self._plot_comprehensive_metrics()
        
        # 2. Combined F1 and Accuracy bar chart
        self._plot_combined_f1_accuracy()
        
        # 3. Comprehensive classifier performance comparison
        self._plot_classifier_performance()
        
        # 4. Individual vectorization technique performance
        self._plot_vectorization_performance()
        
        # 5. Feature importance (for Random Forest)
        self._plot_feature_importance()
        
        # 6. Confusion matrix for best classifier
        self._plot_confusion_matrix()
        
        # 7. Performance summary table
        self._create_performance_summary()
        
        logger.info("Visualizations created and saved")
    
    def _plot_comprehensive_metrics(self) -> None:
        """Create a comprehensive chart showing all four metrics (accuracy, F1-score, precision, recall) for all combinations."""
        # Prepare data for plotting
        data = []
        for rep_name, classifiers in self.classifier_results.items():
            for clf_name, results in classifiers.items():
                # Get all four metrics
                accuracy = results['test_results']['accuracy']
                f1_score = results['test_results']['classification_report']['weighted avg']['f1-score']
                precision = results['test_results']['classification_report']['weighted avg']['precision']
                recall = results['test_results']['classification_report']['weighted avg']['recall']
                
                # Create a combined label
                combined_label = f"{clf_name.replace('_', ' ').title()} - {rep_name.replace('_', ' ').title()}"
                
                data.append({
                    'Combination': combined_label,
                    'Classifier': clf_name.replace('_', ' ').title(),
                    'Vectorization': rep_name.replace('_', ' ').title(),
                    'Accuracy': accuracy,
                    'F1-Score': f1_score,
                    'Precision': precision,
                    'Recall': recall
                })
        
        df = pd.DataFrame(data)
        
        # Create a single figure with grouped bars for all 4 metrics
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Set up the bar positions
        x = np.arange(len(df))
        width = 0.2  # Make bars narrower to fit 4 metrics
        
        # Create bars for all 4 metrics
        bars1 = ax.bar(x - 1.5*width, df['Accuracy'], width, label='Accuracy', 
                      color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        bars2 = ax.bar(x - 0.5*width, df['F1-Score'], width, label='F1-Score', 
                      color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
        bars3 = ax.bar(x + 0.5*width, df['Precision'], width, label='Precision', 
                      color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
        bars4 = ax.bar(x + 1.5*width, df['Recall'], width, label='Recall', 
                      color='gold', alpha=0.8, edgecolor='orange', linewidth=1)
        
        # Customize the plot
        ax.set_xlabel('Classifier - Vectorization Combination', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Comprehensive Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Combination'], rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        add_value_labels(bars4)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comprehensive_metrics_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        df.to_csv(f"{self.results_dir}/comprehensive_metrics_data.csv", index=False)
        
        logger.info("Comprehensive metrics chart created and saved")
    
    def _plot_combined_f1_accuracy(self) -> None:
        """Create a single bar chart showing both F1-score and accuracy for all combinations."""
        # Prepare data for plotting
        data = []
        for rep_name, classifiers in self.classifier_results.items():
            for clf_name, results in classifiers.items():
                # Get both accuracy and F1-score
                accuracy = results['test_results']['accuracy']
                f1_score = results['test_results']['classification_report']['weighted avg']['f1-score']
                
                # Create a combined label
                combined_label = f"{clf_name.replace('_', ' ').title()} - {rep_name.replace('_', ' ').title()}"
                
                data.append({
                    'Combination': combined_label,
                    'Classifier': clf_name.replace('_', ' ').title(),
                    'Vectorization': rep_name.replace('_', ' ').title(),
                    'Accuracy': accuracy,
                    'F1-Score': f1_score
                })
        
        df = pd.DataFrame(data)
        
        # Create a single figure with grouped bars
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Set up the bar positions
        x = np.arange(len(df))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, df['Accuracy'], width, label='Accuracy', 
                      color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        bars2 = ax.bar(x + width/2, df['F1-Score'], width, label='F1-Score', 
                      color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        # Customize the plot
        ax.set_xlabel('Classifier - Vectorization Combination', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Combined F1-Score and Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Combination'], rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/combined_f1_accuracy_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        df.to_csv(f"{self.results_dir}/combined_f1_accuracy_data.csv", index=False)
        
        logger.info("Combined F1-Accuracy chart created and saved")
    
    def _plot_classifier_performance(self) -> None:
        """Plot comprehensive classifier performance comparison across all vectorization techniques."""
        # Prepare data for plotting
        data = []
        for rep_name, classifiers in self.classifier_results.items():
            for clf_name, results in classifiers.items():
                # Get both accuracy and F1-score
                accuracy = results['test_results']['accuracy']
                f1_score = results['test_results']['classification_report']['weighted avg']['f1-score']
                
                data.append({
                    'Vectorization': rep_name.replace('_', ' ').title(),
                    'Classifier': clf_name.replace('_', ' ').title(),
                    'Accuracy': accuracy,
                    'F1-Score': f1_score
                })
        
        df = pd.DataFrame(data)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Accuracy
        sns.barplot(data=df, x='Vectorization', y='Accuracy', hue='Classifier', ax=ax1)
        ax1.set_title('Classifier Performance - Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Vectorization Method', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: F1-Score
        sns.barplot(data=df, x='Vectorization', y='F1-Score', hue='Classifier', ax=ax2)
        ax2.set_title('Classifier Performance - F1-Score', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Vectorization Method', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comprehensive_classifier_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        df.to_csv(f"{self.results_dir}/classifier_performance_data.csv", index=False)
    
    def _plot_vectorization_performance(self) -> None:
        """Plot performance comparison for each vectorization technique."""
        # Prepare data
        data = []
        for rep_name, classifiers in self.classifier_results.items():
            for clf_name, results in classifiers.items():
                accuracy = results['test_results']['accuracy']
                f1_score = results['test_results']['classification_report']['weighted avg']['f1-score']
                precision = results['test_results']['classification_report']['weighted avg']['precision']
                recall = results['test_results']['classification_report']['weighted avg']['recall']
                
                data.append({
                    'Vectorization': rep_name.replace('_', ' ').title(),
                    'Classifier': clf_name.replace('_', ' ').title(),
                    'Accuracy': accuracy,
                    'F1-Score': f1_score,
                    'Precision': precision,
                    'Recall': recall
                })
        
        df = pd.DataFrame(data)
        
        # Create subplots for each metric
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            sns.barplot(data=df, x='Vectorization', y=metric, hue='Classifier', ax=axes[i])
            axes[i].set_title(f'{metric} by Vectorization Method and Classifier', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Vectorization Method', fontsize=12)
            axes[i].set_ylabel(metric, fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/vectorization_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self) -> None:
        """Plot feature importance for Random Forest across different vectorization techniques."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        for i, rep_name in enumerate(['tfidf', 'doc2vec', 'sentence_bert']):
            if rep_name in self.classifiers and 'random_forest' in self.classifiers[rep_name]:
                rf_classifier = self.classifiers[rep_name]['random_forest']
                importance_df = rf_classifier.get_feature_importance_df()
                
                if importance_df is not None:
                    top_features = importance_df.head(15)
                    sns.barplot(data=top_features, x='importance', y='feature', ax=axes[i])
                    axes[i].set_title(f'Top 15 Feature Importance - {rep_name.upper()}', fontsize=14, fontweight='bold')
                    axes[i].set_xlabel('Importance', fontsize=12)
                    axes[i].set_ylabel('Features', fontsize=12)
                    axes[i].grid(axis='x', alpha=0.3)
                else:
                    axes[i].text(0.5, 0.5, f'No feature importance\navailable for {rep_name}', 
                               ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
            else:
                axes[i].text(0.5, 0.5, f'Random Forest not available\nfor {rep_name}', 
                           ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self) -> None:
        """Plot confusion matrix for best classifier across all vectorization techniques."""
        # Find best classifier for each vectorization technique
        best_classifiers = {}
        
        for rep_name, classifiers in self.classifier_results.items():
            best_accuracy = 0
            best_clf_name = None
            
            for clf_name, results in classifiers.items():
                if results['test_results']['accuracy'] > best_accuracy:
                    best_accuracy = results['test_results']['accuracy']
                    best_clf_name = clf_name
            
            if best_clf_name:
                best_classifiers[rep_name] = best_clf_name
        
        # Create subplots for confusion matrices
        n_techniques = len(best_classifiers)
        fig, axes = plt.subplots(1, n_techniques, figsize=(6*n_techniques, 6))
        
        if n_techniques == 1:
            axes = [axes]
        
        # Map representation names to their test attribute names
        rep_to_test_attr = {
            'tfidf': 'tfidf_test',
            'doc2vec': 'doc2vec_test',
            'sentence_bert': 'bert_test'
        }
        
        for i, (rep_name, clf_name) in enumerate(best_classifiers.items()):
            classifier = self.classifiers[rep_name][clf_name]
            test_attr_name = rep_to_test_attr.get(rep_name, f"{rep_name}_test")
            X_test_rep = getattr(self, test_attr_name)
            cm = confusion_matrix(self.y_test, classifier.predict(X_test_rep))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{clf_name.upper()} on {rep_name.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('True Label', fontsize=12)
            axes[i].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_summary(self) -> None:
        """Create a performance summary table."""
        summary_data = []
        
        for rep_name, classifiers in self.classifier_results.items():
            for clf_name, results in classifiers.items():
                summary_data.append({
                    'Vectorization': rep_name.replace('_', ' ').title(),
                    'Classifier': clf_name.replace('_', ' ').title(),
                    'Accuracy': f"{results['test_results']['accuracy']:.4f}",
                    'F1-Score': f"{results['test_results']['classification_report']['weighted avg']['f1-score']:.4f}",
                    'Precision': f"{results['test_results']['classification_report']['weighted avg']['precision']:.4f}",
                    'Recall': f"{results['test_results']['classification_report']['weighted avg']['recall']:.4f}"
                })
        
        # Create summary table
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        df.to_csv(f"{self.results_dir}/performance_summary.csv", index=False)
        
        # Create a formatted table visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Performance Summary - All Classifiers and Vectorization Techniques', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{self.results_dir}/performance_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_pipeline(self) -> None:
        """
        Run the complete pipeline.
        """
        logger.info("Starting complete pipeline...")
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Generate document representations
            self.generate_tfidf_vectors()
            self.generate_doc2vec_vectors()
            self.generate_gemini_summaries_and_bert_vectors()
            
            # Step 3: Train classifiers
            self.train_classifiers()
            
            # Step 4: Generate Gemini reasoning
            gemini_analysis = self.generate_gemini_reasoning()
            
            # Step 5: Create visualizations
            self.create_visualizations()
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """
    Main function to run the pipeline.
    """
    # Initialize pipeline with all 20 newsgroups classes
    pipeline = NewsgroupsPipeline(
        categories=None,  
        max_samples_per_category=200,  # Limit for faster processing
        test_size=0.2,
        random_state=42
    )
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main() 