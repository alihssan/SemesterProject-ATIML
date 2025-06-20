#!/usr/bin/env python3
"""
XAI Analysis Pipeline for Text Classification
Uses LIME for explainability and works with both 20 Newsgroups and AG News datasets.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.newsgroups_dataset import NewsgroupsDataset
from dataset.ag_news_dataset import AGNewsDataset
from classifiers.classifier import Classifier
from vectorization.tf_idf import TFIDFVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class XAIExplanation:
    """Data class to store XAI explanations."""
    document_id: str
    text: str
    true_label: str
    predicted_label: str
    confidence: float
    lime_explanation: Optional[Dict] = None
    llm_explanation: Optional[str] = None
    llm_with_lime_explanation: Optional[str] = None
    agreement_score: Optional[float] = None

class XAIAnalysisPipeline:
    """Pipeline for XAI analysis using LIME with LLM explanations."""
    
    def __init__(self, 
                 dataset_type: str = 'newsgroups',  # 'newsgroups' or 'ag_news'
                 results_dir: str = None,
                 random_state: int = 42,
                 num_samples: int = 10):
        """
        Initialize the XAI Analysis Pipeline.
        
        Args:
            dataset_type: Type of dataset ('newsgroups' or 'ag_news')
            results_dir: Directory to save results
            random_state: Random seed for reproducibility
            num_samples: Number of samples to analyze
        """
        self.dataset_type = dataset_type
        self.random_state = random_state
        self.num_samples = num_samples
        
        # Setup results directory
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"results/xai_analysis_{dataset_type}_{timestamp}"
        else:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.dataset = None
        self.classifier = None
        self.vectorizer = None
        self.explanations = []
        
        # Initialize XAI tools
        self._setup_xai_tools()
        
        # Initialize LLM
        self._setup_llm()
        
        logger.info(f"XAI Analysis Pipeline initialized for {dataset_type}. Results will be saved to: {self.results_dir}")
    
    def _setup_xai_tools(self):
        """Setup LIME explainer."""
        try:
            import lime
            import lime.lime_text
            from lime.lime_text import LimeTextExplainer
            
            # Setup LIME
            self.lime_explainer = LimeTextExplainer(class_names=None)
            logger.info("LIME explainer initialized successfully")
            
        except ImportError as e:
            logger.warning(f"LIME not available: {e}")
            self.lime_explainer = None
    
    def _setup_llm(self):
        """Setup LLM for natural language explanations using Ollama with Gemma 3B."""
        try:
            from vectorization.llm_summarisation import OllamaSummarizer
            
            # Initialize Ollama with Gemma 3B
            self.llm_model = OllamaSummarizer(model_name="gemma:3b")
            logger.info("Ollama LLM with Gemma 3B initialized successfully")
            
            # Test the connection
            try:
                test_response = self.llm_model.summarize("Test message", max_sentences=1)
                logger.info("Ollama connection test successful")
            except Exception as e:
                logger.warning(f"Ollama connection test failed: {e}")
                logger.info("Please ensure Ollama is running with: ollama serve")
                logger.info("And Gemma 3B is available with: ollama pull gemma:3b")
                
        except Exception as e:
            logger.warning(f"Ollama LLM not available: {e}")
            logger.info("To use LLM explanations, please:")
            logger.info("1. Install Ollama: https://ollama.ai/")
            logger.info("2. Pull Gemma 3B: ollama pull gemma:3b")
            logger.info("3. Start Ollama: ollama serve")
            self.llm_model = None
    
    def load_dataset(self):
        """Load the specified dataset."""
        logger.info(f"Loading {self.dataset_type} dataset...")
        
        try:
            if self.dataset_type == 'newsgroups':
                self.dataset = NewsgroupsDataset()
                self.dataset.load_data()
                
                # Split data
                from sklearn.model_selection import train_test_split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.dataset.texts,
                    self.dataset.labels,
                    test_size=0.2,
                    random_state=self.random_state,
                    stratify=self.dataset.labels
                )
                
                # Get class names
                self.class_names = self.dataset.class_names
                
            elif self.dataset_type == 'ag_news':
                self.dataset = AGNewsDataset(random_state=self.random_state)
                
                # Get train and test data
                self.X_train, self.y_train = self.dataset.get_train_data()
                self.X_test, self.y_test = self.dataset.get_test_data()
                
                # Get class names
                self.class_names = self.dataset.class_names
                
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
            logger.info(f"Dataset loaded: {len(self.X_train)} train, {len(self.X_test)} test samples")
            logger.info(f"Number of classes: {len(self.class_names)}")
            logger.info(f"Class names: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def train_classifier(self):
        """Train a classifier for explanations."""
        logger.info("Training classifier for XAI analysis...")
        
        try:
            # Use TF-IDF + Logistic Regression for interpretability
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            
            # Create pipeline
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.classifier = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
            
            # Train
            self.classifier.fit(self.X_train, self.y_train)
            
            # Evaluate
            from sklearn.metrics import accuracy_score, classification_report
            y_pred = self.classifier.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            logger.info(f"Classifier trained successfully. Accuracy: {accuracy:.4f}")
            
            # Save classification report
            report = classification_report(self.y_test, y_pred, target_names=self.class_names, output_dict=True)
            with open(f"{self.results_dir}/classification_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise
    
    def _get_lime_explanation(self, text: str, num_features: int = 10) -> Dict:
        """Get LIME explanation for a text."""
        if self.lime_explainer is None:
            return None
        
        try:
            def predict_proba(texts):
                return self.classifier.predict_proba(texts)
            
            # Get explanation
            exp = self.lime_explainer.explain_instance(
                text,
                predict_proba,
                num_features=num_features,
                num_samples=100
            )
            
            # Extract important features
            explanation = {
                'features': [],
                'weights': [],
                'intercept': exp.intercept[1] if len(exp.intercept) > 1 else exp.intercept[0]
            }
            
            for feature, weight in exp.as_list():
                explanation['features'].append(feature)
                explanation['weights'].append(weight)
            
            return explanation
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return None
    
    def _get_llm_explanation(self, text: str, predicted_label: str, true_label: str) -> str:
        """Get LLM explanation for classification decision using Gemma 3B."""
        if self.llm_model is None:
            return "LLM not available"
        
        try:
            prompt = f"""
            Analyze this text classification and explain the reasoning:

            Text: {text[:800]}...

            Predicted Category: {predicted_label}
            True Category: {true_label}

            Explain why this text was classified as "{predicted_label}":
            1. What key words or phrases suggest this category?
            2. What linguistic patterns indicate this topic?
            3. Does this classification make sense? Why or why not?

            Provide a clear, concise explanation.
            """
            
            response = self.llm_model.summarize(prompt, max_sentences=4)
            return response
            
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}")
            return f"LLM explanation failed: {str(e)}"
    
    def _get_llm_with_lime_explanation(self, text: str, predicted_label: str, true_label: str, 
                                     lime_explanation: Dict) -> str:
        """Get LLM explanation incorporating LIME insights using Gemma 3B."""
        if self.llm_model is None:
            return "LLM not available"
        
        try:
            # Prepare LIME information
            lime_info = ""
            if lime_explanation:
                lime_features = ", ".join([f'"{f}" (weight: {w:.3f})' for f, w in zip(
                    lime_explanation['features'][:5], lime_explanation['weights'][:5])])
                lime_info += f"LIME identified these important features: {lime_features}. "
            
            prompt = f"""
            Analyze this text classification with LIME insights:

            Text: {text[:800]}...

            Predicted Category: {predicted_label}
            True Category: {true_label}

            LIME Analysis Results:
            {lime_info}

            Provide an explanation that:
            1. Incorporates the LIME insights about important features
            2. Explains whether the LIME analysis supports this classification
            3. Gives your own reasoning about why this text belongs to this category
            4. Discusses if the classification is correct

            Be clear and comprehensive in your analysis.
            """
            
            response = self.llm_model.summarize(prompt, max_sentences=6)
            return response
            
        except Exception as e:
            logger.warning(f"LLM with LIME explanation failed: {e}")
            return f"LLM with LIME explanation failed: {str(e)}"
    
    def _calculate_agreement_score(self, lime_explanation: Dict, llm_explanation: str, 
                                 llm_with_lime: str) -> float:
        """Calculate agreement score between LIME and LLM explanations."""
        try:
            if not lime_explanation:
                return 0.0
            
            # Get LIME features
            lime_features = set(lime_explanation['features'])
            
            if not lime_features:
                return 0.0
            
            # Count features mentioned in LLM explanations
            llm_text = (llm_explanation + " " + llm_with_lime).lower()
            mentioned_features = sum(1 for feature in lime_features if feature.lower() in llm_text)
            
            agreement_score = mentioned_features / len(lime_features)
            return min(agreement_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Agreement score calculation failed: {e}")
            return 0.0
    
    def analyze_samples(self):
        """Analyze selected samples with LIME and LLM explanations."""
        logger.info(f"Analyzing {self.num_samples} samples...")
        
        # Select diverse samples
        np.random.seed(self.random_state)
        sample_indices = np.random.choice(len(self.X_test), self.num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            logger.info(f"Analyzing sample {i+1}/{self.num_samples}")
            
            text = self.X_test[idx]
            true_label = self.y_test[idx]
            predicted_label = self.classifier.predict([text])[0]
            confidence = np.max(self.classifier.predict_proba([text]))
            
            # Get class names
            true_label_name = self.class_names[true_label]
            predicted_label_name = self.class_names[predicted_label]
            
            # Get explanations
            lime_explanation = self._get_lime_explanation(text)
            llm_explanation = self._get_llm_explanation(text, predicted_label_name, true_label_name)
            llm_with_lime = self._get_llm_with_lime_explanation(text, predicted_label_name, true_label_name, 
                                                              lime_explanation)
            
            # Calculate agreement score
            agreement_score = self._calculate_agreement_score(lime_explanation, llm_explanation, llm_with_lime)
            
            # Store explanation
            explanation = XAIExplanation(
                document_id=f"sample_{i+1}",
                text=text,
                true_label=true_label_name,
                predicted_label=predicted_label_name,
                confidence=confidence,
                lime_explanation=lime_explanation,
                llm_explanation=llm_explanation,
                llm_with_lime_explanation=llm_with_lime,
                agreement_score=agreement_score
            )
            
            self.explanations.append(explanation)
    
    def create_visualizations(self):
        """Create visualizations for XAI analysis."""
        logger.info("Creating visualizations...")
        
        # 1. Agreement Score Distribution
        self._plot_agreement_distribution()
        
        # 2. LIME Feature Importance
        self._plot_lime_feature_importance()
        
        # 3. Confidence vs Agreement
        self._plot_confidence_vs_agreement()
        
        # 4. Sample Explanations
        self._create_explanation_samples()
    
    def _plot_agreement_distribution(self):
        """Plot distribution of agreement scores."""
        agreement_scores = [exp.agreement_score for exp in self.explanations]
        
        plt.figure(figsize=(10, 6))
        plt.hist(agreement_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.xlabel('Agreement Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Agreement Scores between LIME and LLM Explanations ({self.dataset_type})')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = np.mean(agreement_scores)
        plt.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/agreement_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Agreement distribution saved. Mean agreement: {mean_score:.3f}")
    
    def _plot_lime_feature_importance(self):
        """Plot LIME feature importance across samples."""
        lime_features = []
        
        for exp in self.explanations:
            if exp.lime_explanation:
                lime_features.extend(exp.lime_explanation['features'][:3])
        
        # Count feature frequencies
        from collections import Counter
        lime_counts = Counter(lime_features)
        
        # Get top features
        top_features = [f for f, c in lime_counts.most_common(15)]
        top_counts = [c for f, c in lime_counts.most_common(15)]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_counts, color='lightcoral', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Frequency')
        plt.title(f'Top LIME Features ({self.dataset_type})')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, top_counts)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/lime_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("LIME feature importance plot saved")
    
    def _plot_confidence_vs_agreement(self):
        """Plot confidence vs agreement score."""
        confidences = [exp.confidence for exp in self.explanations]
        agreement_scores = [exp.agreement_score for exp in self.explanations]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(confidences, agreement_scores, alpha=0.7, s=100)
        plt.xlabel('Classification Confidence')
        plt.ylabel('Agreement Score')
        plt.title(f'Classification Confidence vs LIME-LLM Agreement ({self.dataset_type})')
        plt.grid(True, alpha=0.3)
        
        # Add correlation line
        correlation = np.corrcoef(confidences, agreement_scores)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/confidence_vs_agreement.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence vs agreement plot saved. Correlation: {correlation:.3f}")
    
    def _create_explanation_samples(self):
        """Create detailed explanation samples for manual inspection."""
        logger.info("Creating detailed explanation samples...")
        
        # Select samples with different agreement levels
        high_agreement = [exp for exp in self.explanations if exp.agreement_score > 0.7]
        low_agreement = [exp for exp in self.explanations if exp.agreement_score < 0.3]
        medium_agreement = [exp for exp in self.explanations if 0.3 <= exp.agreement_score <= 0.7]
        
        samples_to_show = []
        if high_agreement:
            samples_to_show.append(("High Agreement", high_agreement[0]))
        if low_agreement:
            samples_to_show.append(("Low Agreement", low_agreement[0]))
        if medium_agreement:
            samples_to_show.append(("Medium Agreement", medium_agreement[0]))
        
        # Create detailed report
        report = []
        report.append(f"XAI Analysis Report for {self.dataset_type.upper()} Dataset")
        report.append("=" * 80)
        report.append("")
        
        for category, exp in samples_to_show:
            report.append(f"=== {category} Sample (Agreement: {exp.agreement_score:.3f}) ===\n")
            report.append(f"Document ID: {exp.document_id}")
            report.append(f"True Label: {exp.true_label}")
            report.append(f"Predicted Label: {exp.predicted_label}")
            report.append(f"Confidence: {exp.confidence:.3f}")
            report.append(f"Correct: {exp.true_label == exp.predicted_label}\n")
            
            report.append("Text (first 500 chars):")
            report.append(f"{exp.text[:500]}...\n")
            
            if exp.lime_explanation:
                report.append("LIME Explanation:")
                for feature, weight in zip(exp.lime_explanation['features'][:5], 
                                         exp.lime_explanation['weights'][:5]):
                    report.append(f"  {feature}: {weight:.3f}")
                report.append("")
            
            report.append("LLM Explanation:")
            report.append(exp.llm_explanation)
            report.append("")
            
            report.append("LLM with LIME Explanation:")
            report.append(exp.llm_with_lime_explanation)
            report.append("\n" + "="*80 + "\n")
        
        # Save report
        with open(f"{self.results_dir}/detailed_explanations.txt", 'w') as f:
            f.write('\n'.join(report))
        
        logger.info("Detailed explanation samples saved")
    
    def save_results(self):
        """Save all results to files."""
        logger.info("Saving results...")
        
        # Save explanations as JSON
        explanations_data = []
        for exp in self.explanations:
            exp_dict = {
                'document_id': exp.document_id,
                'text': exp.text,
                'true_label': exp.true_label,
                'predicted_label': exp.predicted_label,
                'confidence': exp.confidence,
                'lime_explanation': exp.lime_explanation,
                'llm_explanation': exp.llm_explanation,
                'llm_with_lime_explanation': exp.llm_with_lime_explanation,
                'agreement_score': exp.agreement_score
            }
            explanations_data.append(exp_dict)
        
        with open(f"{self.results_dir}/explanations.json", 'w') as f:
            json.dump(explanations_data, f, indent=2)
        
        # Save summary statistics
        summary = {
            'dataset_type': self.dataset_type,
            'total_samples': len(self.explanations),
            'mean_agreement_score': np.mean([exp.agreement_score for exp in self.explanations]),
            'std_agreement_score': np.std([exp.agreement_score for exp in self.explanations]),
            'mean_confidence': np.mean([exp.confidence for exp in self.explanations]),
            'accuracy': np.mean([exp.true_label == exp.predicted_label for exp in self.explanations]),
            'lime_available': sum(1 for exp in self.explanations if exp.lime_explanation is not None),
            'llm_available': sum(1 for exp in self.explanations if exp.llm_explanation != "LLM not available")
        }
        
        with open(f"{self.results_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as pickle for further analysis
        with open(f"{self.results_dir}/xai_analysis_results.pkl", 'wb') as f:
            pickle.dump(self.explanations, f)
        
        logger.info("Results saved successfully")
    
    def run_complete_analysis(self):
        """Run the complete XAI analysis pipeline."""
        logger.info(f"Starting XAI Analysis Pipeline for {self.dataset_type}...")
        
        try:
            # Step 1: Load dataset
            self.load_dataset()
            
            # Step 2: Train classifier
            self.train_classifier()
            
            # Step 3: Analyze samples
            self.analyze_samples()
            
            # Step 4: Create visualizations
            self.create_visualizations()
            
            # Step 5: Save results
            self.save_results()
            
            logger.info("XAI Analysis Pipeline completed successfully!")
            logger.info(f"Results saved to: {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the XAI analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='XAI Analysis Pipeline')
    parser.add_argument('--dataset', choices=['newsgroups', 'ag_news'], 
                       default='newsgroups', help='Dataset to analyze')
    parser.add_argument('--samples', type=int, default=15, 
                       help='Number of samples to analyze')
    parser.add_argument('--random_state', type=int, default=42, 
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = XAIAnalysisPipeline(
        dataset_type=args.dataset,
        num_samples=args.samples,
        random_state=args.random_state
    )
    pipeline.run_complete_analysis()

if __name__ == "__main__":
    main() 