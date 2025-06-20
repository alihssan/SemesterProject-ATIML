"""
AG News Pipeline with Zero-Shot and Few-Shot Learning

This pipeline implements:
1. Zero-shot classification using pre-trained language models
2. Few-shot learning with different shot counts
3. Traditional supervised learning for comparison
4. Prompt engineering for better zero/few-shot performance
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.ag_news_dataset import AGNewsDataset
from config.config import load_config
from classifiers.classifier import UnifiedClassifier, train_test_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ag_news_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - zero-shot capabilities limited")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - GPT-based zero-shot disabled")

class AGNewsPipeline:
    """
    Pipeline for AG News dataset with zero-shot and few-shot learning capabilities.
    """
    
    def __init__(self, 
                 config_mode: str = 'ag_news',
                 max_samples_per_class: int = 1000,
                 few_shot_shots: List[int] = [1, 3, 5, 10],
                 random_state: int = 42):
        """
        Initialize the AG News pipeline.
        
        Args:
            config_mode: Configuration mode
            max_samples_per_class: Maximum samples per class for faster processing
            few_shot_shots: List of shot counts for few-shot learning
            random_state: Random seed for reproducibility
        """
        self.config_mode = config_mode
        self.max_samples_per_class = max_samples_per_class
        self.few_shot_shots = few_shot_shots
        self.random_state = random_state
        
        # Load configuration
        self.config = load_config(config_mode)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results/ag_news_pipeline_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize results storage
        self.dataset = None
        self.zero_shot_results = None
        self.few_shot_results = {}
        self.few_shot_shots = [1, 2, 5, 10]  # Number of shots per class
        
        # Initialize zero-shot models
        self.zero_shot_classifier = None
        self.sentence_transformer = None
        
        logger.info(f"AG News Pipeline initialized. Results will be saved to: {self.results_dir}")
    
    def load_dataset(self) -> None:
        """Load and prepare the AG News dataset."""
        logger.info("Loading AG News dataset...")
        
        try:
            # Create dataset with few-shot configurations
            self.dataset = AGNewsDataset(
                random_state=self.random_state,
                test_size=0.2,
                val_size=0.1,
                few_shot_shots=self.few_shot_shots
            )
            
            # Get balanced subset for faster processing
            if self.max_samples_per_class:
                X_train, y_train = self.dataset.get_train_data()
                X_test, y_test = self.dataset.get_test_data()
                
                # Create balanced subset
                train_df = pd.DataFrame({'text': X_train, 'label': y_train})
                test_df = pd.DataFrame({'text': X_test, 'label': y_test})
                
                # Sample balanced subset
                train_balanced = []
                test_balanced = []
                
                for class_idx in range(self.dataset.num_classes):
                    class_train = train_df[train_df['label'] == class_idx]
                    class_test = test_df[test_df['label'] == class_idx]
                    
                    # Sample from train
                    if len(class_train) > self.max_samples_per_class:
                        class_train = class_train.sample(n=self.max_samples_per_class, random_state=self.random_state)
                    
                    # Sample from test (keep more for evaluation)
                    test_samples = min(self.max_samples_per_class // 2, len(class_test))
                    if len(class_test) > test_samples:
                        class_test = class_test.sample(n=test_samples, random_state=self.random_state)
                    
                    train_balanced.append(class_train)
                    test_balanced.append(class_test)
                
                # Combine and update dataset
                train_balanced_df = pd.concat(train_balanced, ignore_index=True)
                test_balanced_df = pd.concat(test_balanced, ignore_index=True)
                
                self.X_train = train_balanced_df['text'].tolist()
                self.y_train = train_balanced_df['label'].values
                self.X_test = test_balanced_df['text'].tolist()
                self.y_test = test_balanced_df['label'].values
                
                logger.info(f"Created balanced subset: {len(self.X_train)} train, {len(self.X_test)} test samples")
            else:
                # Use full dataset
                self.X_train, self.y_train = self.dataset.get_train_data()
                self.X_test, self.y_test = self.dataset.get_test_data()
            
            # Save dataset info
            self._save_dataset_info()
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _save_dataset_info(self) -> None:
        """Save dataset information."""
        dataset_info = {
            'num_classes': self.dataset.num_classes,
            'class_names': self.dataset.class_names,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'few_shot_shots': self.few_shot_shots,
            'class_distribution_train': dict(pd.Series(self.y_train).value_counts().sort_index()),
            'class_distribution_test': dict(pd.Series(self.y_test).value_counts().sort_index())
        }
        
        with open(f"{self.results_dir}/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info("Dataset information saved")
    
    def setup_zero_shot_classifier(self) -> None:
        """Set up zero-shot classification models."""
        logger.info("Setting up zero-shot classifier...")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - skipping zero-shot setup")
            return
        
        try:
            # Use a pre-trained model for zero-shot classification
            model_name = "facebook/bart-large-mnli"  # Good for zero-shot classification
            
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Also set up sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Zero-shot classifier set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up zero-shot classifier: {e}")
            self.zero_shot_classifier = None
    
    def run_zero_shot_classification(self) -> None:
        """Run zero-shot classification on test data."""
        if self.zero_shot_classifier is None:
            logger.warning("Zero-shot classifier not available")
            return
        
        logger.info("Running zero-shot classification...")
        
        # Define candidate labels (class names)
        candidate_labels = self.dataset.class_names
        
        # Create prompts for better zero-shot performance
        prompts = [
            "This news article is about:",
            "The topic of this text is:",
            "This article discusses:",
            "The subject matter is:"
        ]
        
        results = {
            'predictions': [],
            'confidences': [],
            'true_labels': self.y_test.tolist(),
            'prompt_used': prompts[0]  # Use first prompt for now
        }
        
        # Process test samples
        for i, text in enumerate(self.X_test):
            if i % 100 == 0:
                logger.info(f"Processing sample {i+1}/{len(self.X_test)}")
            
            try:
                # Run zero-shot classification
                result = self.zero_shot_classifier(
                    text,
                    candidate_labels,
                    hypothesis_template=prompts[0]
                )
                
                # Get prediction and confidence
                predicted_label = result['labels'][0]
                confidence = result['scores'][0]
                
                # Convert label name to index
                predicted_idx = candidate_labels.index(predicted_label)
                
                results['predictions'].append(predicted_idx)
                results['confidences'].append(confidence)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                # Default prediction
                results['predictions'].append(0)
                results['confidences'].append(0.0)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(results['true_labels'], results['predictions'])
        report = classification_report(
            results['true_labels'], 
            results['predictions'], 
            target_names=candidate_labels,
            output_dict=True
        )
        
        self.zero_shot_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': results['predictions'],
            'confidences': results['confidences'],
            'true_labels': results['true_labels']
        }
        
        logger.info(f"Zero-shot classification completed. Accuracy: {accuracy:.4f}")
        
        # Save results
        with open(f"{self.results_dir}/zero_shot_results.pkl", 'wb') as f:
            pickle.dump(self.zero_shot_results, f)
    
    def run_few_shot_classification(self) -> None:
        """Run few-shot classification for different shot counts."""
        logger.info("Running few-shot classification...")
        
        for shots in self.few_shot_shots:
            logger.info(f"Running {shots}-shot classification...")
            
            try:
                # Get few-shot data
                few_shot_data = self.dataset.get_few_shot_data(shots)
                X_train_few = few_shot_data['X_train']
                y_train_few = few_shot_data['y_train']
                X_test_few = few_shot_data['X_test']
                y_test_few = few_shot_data['y_test']
                
                # Use sentence transformer for embeddings
                if self.sentence_transformer is not None:
                    # Get embeddings
                    train_embeddings = self.sentence_transformer.encode(X_train_few)
                    test_embeddings = self.sentence_transformer.encode(X_test_few)
                    
                    # Train simple classifier on embeddings
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import accuracy_score, classification_report
                    
                    clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
                    clf.fit(train_embeddings, y_train_few)
                    
                    # Predict
                    predictions = clf.predict(test_embeddings)
                    accuracy = accuracy_score(y_test_few, predictions)
                    report = classification_report(
                        y_test_few, predictions,
                        target_names=self.dataset.class_names,
                        output_dict=True
                    )
                    
                    self.few_shot_results[f'{shots}_shot'] = {
                        'accuracy': accuracy,
                        'classification_report': report,
                        'predictions': predictions.tolist(),
                        'true_labels': y_test_few.tolist(),
                        'train_samples': len(X_train_few),
                        'test_samples': len(X_test_few)
                    }
                    
                    logger.info(f"{shots}-shot accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error in {shots}-shot classification: {e}")
        
        # Save results
        with open(f"{self.results_dir}/few_shot_results.pkl", 'wb') as f:
            pickle.dump(self.few_shot_results, f)
    
    def create_visualizations(self) -> None:
        """Create visualizations comparing zero-shot, few-shot, and supervised results."""
        logger.info("Creating visualizations...")
        
        # 1. Comprehensive metrics chart (all 4 metrics)
        self._plot_comprehensive_metrics()
        
        # 2. Combined F1 and Accuracy chart
        self._plot_combined_f1_accuracy()
        
        # 3. Learning curve analysis
        self._plot_learning_curves()
        
        # 4. Performance summary table
        self._create_performance_summary()
        
        logger.info("Visualizations created and saved")
    
    def _plot_comprehensive_metrics(self) -> None:
        """Create a comprehensive chart showing all four metrics (accuracy, F1-score, precision, recall) for all methods."""
        # Prepare data for plotting
        data = []
        
        # Add zero-shot results
        if self.zero_shot_results:
            data.append({
                'Method': 'Zero-Shot',
                'Accuracy': self.zero_shot_results['accuracy'],
                'F1-Score': self.zero_shot_results['classification_report']['weighted avg']['f1-score'],
                'Precision': self.zero_shot_results['classification_report']['weighted avg']['precision'],
                'Recall': self.zero_shot_results['classification_report']['weighted avg']['recall'],
                'Shots': 0
            })
        
        # Add few-shot results
        for shots in self.few_shot_shots:
            key = f'{shots}_shot'
            if key in self.few_shot_results:
                data.append({
                    'Method': f'{shots}-Shot',
                    'Accuracy': self.few_shot_results[key]['accuracy'],
                    'F1-Score': self.few_shot_results[key]['classification_report']['weighted avg']['f1-score'],
                    'Precision': self.few_shot_results[key]['classification_report']['weighted avg']['precision'],
                    'Recall': self.few_shot_results[key]['classification_report']['weighted avg']['recall'],
                    'Shots': shots
                })
        
        # Note: Supervised results are not included in this run
        
        df = pd.DataFrame(data)
        
        # Create a single figure with grouped bars for all 4 metrics
        fig, ax = plt.subplots(figsize=(16, 10))
        
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
        ax.set_xlabel('Learning Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('AG News: Zero-Shot vs Few-Shot Learning - Comprehensive Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Method'], rotation=45, ha='right', fontsize=12)
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
        plt.savefig(f"{self.results_dir}/ag_news_comprehensive_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        df.to_csv(f"{self.results_dir}/ag_news_comprehensive_metrics.csv", index=False)
        
        logger.info("Comprehensive metrics chart created and saved")
    
    def _plot_combined_f1_accuracy(self) -> None:
        """Create a combined F1-score and accuracy chart."""
        # Prepare data for plotting
        data = []
        
        # Add zero-shot results
        if self.zero_shot_results:
            data.append({
                'Method': 'Zero-Shot',
                'Accuracy': self.zero_shot_results['accuracy'],
                'F1-Score': self.zero_shot_results['classification_report']['weighted avg']['f1-score'],
                'Shots': 0
            })
        
        # Add few-shot results
        for shots in self.few_shot_shots:
            key = f'{shots}_shot'
            if key in self.few_shot_results:
                data.append({
                    'Method': f'{shots}-Shot',
                    'Accuracy': self.few_shot_results[key]['accuracy'],
                    'F1-Score': self.few_shot_results[key]['classification_report']['weighted avg']['f1-score'],
                    'Shots': shots
                })
        
        # Note: Supervised results are not included in this run
        
        df = pd.DataFrame(data)
        
        # Create combined chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['Accuracy'], width, label='Accuracy', 
                      color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        bars2 = ax.bar(x + width/2, df['F1-Score'], width, label='F1-Score', 
                      color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        ax.set_xlabel('Learning Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('AG News: Zero-Shot vs Few-Shot Learning', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Method'], rotation=45, ha='right', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/ag_news_comparison_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data
        df.to_csv(f"{self.results_dir}/ag_news_comparison_data.csv", index=False)
        
        logger.info("Combined F1-Accuracy chart created and saved")
    
    def _plot_learning_curves(self) -> None:
        """Create learning curves showing performance vs number of shots."""
        # Prepare data for learning curves
        shots_data = []
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        # Add zero-shot (0 shots)
        if self.zero_shot_results:
            shots_data.append({
                'Shots': 0,
                'Accuracy': self.zero_shot_results['accuracy'],
                'F1-Score': self.zero_shot_results['classification_report']['weighted avg']['f1-score'],
                'Precision': self.zero_shot_results['classification_report']['weighted avg']['precision'],
                'Recall': self.zero_shot_results['classification_report']['weighted avg']['recall']
            })
        
        # Add few-shot results
        for shots in self.few_shot_shots:
            key = f'{shots}_shot'
            if key in self.few_shot_results:
                shots_data.append({
                    'Shots': shots,
                    'Accuracy': self.few_shot_results[key]['accuracy'],
                    'F1-Score': self.few_shot_results[key]['classification_report']['weighted avg']['f1-score'],
                    'Precision': self.few_shot_results[key]['classification_report']['weighted avg']['precision'],
                    'Recall': self.few_shot_results[key]['classification_report']['weighted avg']['recall']
                })
        
        # Note: Supervised results are not included in this run
        
        df_curves = pd.DataFrame(shots_data)
        
        # Create learning curves
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i]
            
            # Plot learning curve
            ax.plot(df_curves['Shots'], df_curves[metric], 
                   marker='o', linewidth=2, markersize=8, color=color, alpha=0.8)
            
            # Add data points
            for idx, row in df_curves.iterrows():
                ax.annotate(f'{row[metric]:.3f}', 
                           (row['Shots'], row[metric]),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=9, fontweight='bold')
            
            ax.set_title(f'{metric} Learning Curve (Zero-Shot to Few-Shot)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Shots per Class', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            # Add baseline
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/ag_news_learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save learning curves data
        df_curves.to_csv(f"{self.results_dir}/ag_news_learning_curves.csv", index=False)
        
        logger.info("Learning curves created and saved")
    
    def _create_performance_summary(self) -> None:
        """Create a comprehensive performance summary table."""
        logger.info("Creating performance summary...")
        
        # Prepare summary data
        summary_data = []
        
        # Add zero-shot results
        if self.zero_shot_results:
            summary_data.append({
                'Method': 'Zero-Shot',
                'Shots_per_Class': 0,
                'Total_Training_Samples': 0,
                'Accuracy': f"{self.zero_shot_results['accuracy']:.4f}",
                'F1-Score': f"{self.zero_shot_results['classification_report']['weighted avg']['f1-score']:.4f}",
                'Precision': f"{self.zero_shot_results['classification_report']['weighted avg']['precision']:.4f}",
                'Recall': f"{self.zero_shot_results['classification_report']['weighted avg']['recall']:.4f}",
                'Test_Samples': len(self.zero_shot_results['true_labels'])
            })
        
        # Add few-shot results
        for shots in self.few_shot_shots:
            key = f'{shots}_shot'
            if key in self.few_shot_results:
                summary_data.append({
                    'Method': f'{shots}-Shot',
                    'Shots_per_Class': shots,
                    'Total_Training_Samples': self.few_shot_results[key]['train_samples'],
                    'Accuracy': f"{self.few_shot_results[key]['accuracy']:.4f}",
                    'F1-Score': f"{self.few_shot_results[key]['classification_report']['weighted avg']['f1-score']:.4f}",
                    'Precision': f"{self.few_shot_results[key]['classification_report']['weighted avg']['precision']:.4f}",
                    'Recall': f"{self.few_shot_results[key]['classification_report']['weighted avg']['recall']:.4f}",
                    'Test_Samples': self.few_shot_results[key]['test_samples']
                })
        
        # Note: Supervised results are not included in this run
        
        # Create summary DataFrame
        df_summary = pd.DataFrame(summary_data)
        
        # Save to CSV
        df_summary.to_csv(f"{self.results_dir}/ag_news_performance_summary.csv", index=False)
        
        # Create a formatted table visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(df_summary.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('AG News: Zero-Shot vs Few-Shot Learning - Performance Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{self.results_dir}/ag_news_performance_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        detailed_results = {
            'zero_shot': self.zero_shot_results,
            'few_shot': self.few_shot_results,
            'dataset_info': {
                'num_classes': self.dataset.num_classes,
                'class_names': self.dataset.class_names,
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test)
            }
        }
        
        with open(f"{self.results_dir}/ag_news_detailed_results.pkl", 'wb') as f:
            pickle.dump(detailed_results, f)
        
        logger.info("Performance summary created and saved")
    
    def run_complete_pipeline(self) -> None:
        """Run the complete AG News pipeline (Zero-shot and Few-shot only)."""
        logger.info("Starting AG News pipeline (Zero-shot and Few-shot only)...")
        
        try:
            # Step 1: Load dataset
            self.load_dataset()
            
            # Step 2: Setup zero-shot classifier
            self.setup_zero_shot_classifier()
            
            # Step 3: Run zero-shot classification
            self.run_zero_shot_classification()
            
            # Step 4: Run few-shot classification
            self.run_few_shot_classification()
            
            # Step 5: Create visualizations (without supervised results)
            self.create_visualizations()
            
            logger.info("AG News pipeline completed successfully!")
            logger.info(f"Results saved to: {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the AG News pipeline."""
    # Initialize pipeline
    pipeline = AGNewsPipeline(
        config_mode='ag_news',
        max_samples_per_class=500,  # Limit for faster processing
        few_shot_shots=[1, 3, 5, 10],
        random_state=42
    )
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main() 