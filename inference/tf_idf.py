"""
TF-IDF Inference Module

This module provides inference capabilities for TF-IDF-based text classification.
It can load trained models and make predictions on new text data.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
import warnings
from typing import Union, List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import json

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TFIDFInference:
    """
    TF-IDF Inference class for making predictions on new text data.
    """
    
    def __init__(self, model_path: str = None, tfidf_path: str = None, label_encoder_path: str = None):
        """
        Initialize the TF-IDF inference system.
        
        Args:
            model_path: Path to the trained model pickle file
            tfidf_path: Path to the TF-IDF vectorizer pickle file
            label_encoder_path: Path to the label encoder pickle file
        """
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.model_name = None
        self.feature_names = None
        
        if model_path and tfidf_path and label_encoder_path:
            self.load_model(model_path, tfidf_path, label_encoder_path)
    
    def load_model(self, model_path: str, tfidf_path: str, label_encoder_path: str):
        """
        Load trained model, TF-IDF vectorizer, and label encoder.
        
        Args:
            model_path: Path to the trained model
            tfidf_path: Path to the TF-IDF vectorizer
            label_encoder_path: Path to the label encoder
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Loading TF-IDF vectorizer from: {tfidf_path}")
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            logger.info(f"Loading label encoder from: {label_encoder_path}")
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Determine model type
            if isinstance(self.model, SVC):
                self.model_name = "SVC"
            elif isinstance(self.model, MLPClassifier):
                self.model_name = "MLPClassifier"
            elif isinstance(self.model, SGDClassifier):
                self.model_name = "SGDClassifier"
            elif isinstance(self.model, RandomForestClassifier):
                self.model_name = "RandomForest"
            elif hasattr(self.model, '__class__') and 'XGB' in self.model.__class__.__name__:
                self.model_name = "XGBoost"
            else:
                self.model_name = "Unknown"
            
            logger.info(f"Successfully loaded {self.model_name} model")
            logger.info(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
            logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for inference (same as training preprocessing).
        
        Args:
            text: Raw text input
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Basic preprocessing (same as in NewsgroupsDataset)
        import re
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Download required NLTK data if not available
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back to text
        return ' '.join(tokens)
    
    def predict(self, text: str, return_proba: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Make prediction on a single text input.
        
        Args:
            text: Input text to classify
            return_proba: Whether to return prediction probabilities
            
        Returns:
            Predicted class label or dict with prediction details
        """
        if self.model is None or self.tfidf_vectorizer is None or self.label_encoder is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vector = self.tfidf_vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vector)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        if return_proba:
            try:
                probabilities = self.model.predict_proba(text_vector)[0]
                proba_dict = dict(zip(self.label_encoder.classes_, probabilities))
                return {
                    'predicted_class': predicted_class,
                    'confidence': float(max(probabilities)),
                    'probabilities': proba_dict,
                    'model_name': self.model_name
                }
            except:
                return {
                    'predicted_class': predicted_class,
                    'confidence': None,
                    'probabilities': None,
                    'model_name': self.model_name
                }
        else:
            return predicted_class
    
    def predict_batch(self, texts: List[str], return_proba: bool = False) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Make predictions on a batch of texts.
        
        Args:
            texts: List of input texts to classify
            return_proba: Whether to return prediction probabilities
            
        Returns:
            List of predicted class labels or list of dicts with prediction details
        """
        if self.model is None or self.tfidf_vectorizer is None or self.label_encoder is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        text_vectors = self.tfidf_vectorizer.transform(processed_texts)
        
        # Make predictions
        predictions = self.model.predict(text_vectors)
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        if return_proba:
            try:
                probabilities = self.model.predict_proba(text_vectors)
                results = []
                for i, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
                    proba_dict = dict(zip(self.label_encoder.classes_, probs))
                    results.append({
                        'predicted_class': pred_class,
                        'confidence': float(max(probs)),
                        'probabilities': proba_dict,
                        'model_name': self.model_name
                    })
                return results
            except:
                return [{'predicted_class': pred_class, 'model_name': self.model_name} 
                       for pred_class in predicted_classes]
        else:
            return predicted_classes.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "tfidf_vocabulary_size": len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else None,
            "num_classes": len(self.label_encoder.classes_) if self.label_encoder else None,
            "classes": self.label_encoder.classes_.tolist() if self.label_encoder else None
        }
        
        return info

def save_model_for_inference(model, tfidf_vectorizer, label_encoder, model_name: str, 
                           save_dir: str = "models"):
    """
    Save trained model, TF-IDF vectorizer, and label encoder for inference.
    
    Args:
        model: Trained classifier model
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Fitted label encoder
        model_name: Name for the model files
        save_dir: Directory to save the model files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save TF-IDF vectorizer
    tfidf_path = os.path.join(save_dir, f"{model_name}_tfidf.pkl")
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Save label encoder
    encoder_path = os.path.join(save_dir, f"{model_name}_label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    logger.info(f"Model saved to {save_dir}/")
    logger.info(f"  - Model: {model_path}")
    logger.info(f"  - TF-IDF: {tfidf_path}")
    logger.info(f"  - Label Encoder: {encoder_path}")
    
    return model_path, tfidf_path, encoder_path

# Example usage and testing
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🚀 TF-IDF INFERENCE MODULE TEST")
    logger.info("="*80)
    
    # Create results directory for inference results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_results_dir = f"results/tfidf_inference_{timestamp}"
    os.makedirs(inference_results_dir, exist_ok=True)
    logger.info(f"Created inference results directory: {inference_results_dir}")
    
    # Load Newsgroups test samples for realistic testing
    logger.info("\n📚 Loading Newsgroups test samples...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from dataset.newsgroups_dataset import NewsgroupsDataset
        
        # Initialize dataset
        dataset = NewsgroupsDataset(
            remove_headers=True, 
            remove_footers=True, 
            remove_quotes=True, 
            preprocess=True,
            remove_empty=True,
            clip_long_docs=True,
            clip_percentile=95
        )
        
        # Get test samples
        df_test = dataset.create_dataframe(split='test', raw=False)
        
        # Sample a few examples from different classes
        sample_size = 8
        test_samples = []
        test_labels = []
        
        # Get unique classes
        unique_classes = df_test['class_name'].unique()
        
        for class_name in unique_classes[:4]:  # Take first 4 classes
            class_samples = df_test[df_test['class_name'] == class_name].head(2)
            for _, row in class_samples.iterrows():
                # Convert tokens back to text
                text = ' '.join(row['data'])
                test_samples.append(text)
                test_labels.append(class_name)
        
        logger.info(f"✅ Loaded {len(test_samples)} test samples from Newsgroups dataset")
        logger.info(f"📊 Sample classes: {list(set(test_labels))}")
        
        # Display sample texts
        logger.info("\n📝 Sample texts from test set:")
        for i, (text, label) in enumerate(zip(test_samples, test_labels)):
            logger.info(f"Sample {i+1} ({label}): {text[:100]}...")
        
        test_texts = test_samples
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load Newsgroups test samples: {str(e)}")
        logger.info("📝 Using example texts instead...")
        
        # Fallback example texts
        test_texts = [
            "The new graphics card has amazing performance for gaming",
            "NASA announced plans for a new Mars mission",
            "The baseball game was exciting with a home run in the final inning",
            "The new medical treatment shows promising results in clinical trials"
        ]
        test_labels = ["comp.graphics", "sci.space", "rec.sport.baseball", "sci.med"]
    
    # Check if model files exist
    model_dir = "models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if model_files:
            # Find the most recent model files
            model_name = "tfidf_classifier"  # Default name
            model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
            tfidf_path = os.path.join(model_dir, f"{model_name}_tfidf.pkl")
            encoder_path = os.path.join(model_dir, f"{model_name}_label_encoder.pkl")
            
            if all(os.path.exists(p) for p in [model_path, tfidf_path, encoder_path]):
                logger.info("✅ Found existing model files, testing inference...")
                
                # Initialize inference
                inference = TFIDFInference()
                inference.load_model(model_path, tfidf_path, encoder_path)
                
                # Get model info
                info = inference.get_model_info()
                logger.info(f"🤖 Model Info: {info}")
                
                # Save model info
                with open(f"{inference_results_dir}/model_info.json", 'w') as f:
                    json.dump(info, f, indent=2, default=str)
                logger.info(f"💾 Saved model info to: {inference_results_dir}/model_info.json")
                
                # Test predictions
                logger.info("\n🔍 Testing predictions on Newsgroups test samples:")
                prediction_results = []
                
                for i, (text, true_label) in enumerate(zip(test_texts, test_labels), 1):
                    result = inference.predict(text, return_proba=True)
                    is_correct = result['predicted_class'] == true_label
                    
                    prediction_result = {
                        'sample_id': i,
                        'text': text,
                        'true_label': true_label,
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'is_correct': is_correct,
                        'probabilities': result['probabilities']
                    }
                    prediction_results.append(prediction_result)
                    
                    logger.info(f"📊 Sample {i} (True: {true_label}):")
                    logger.info(f"  📝 Text: {text[:80]}...")
                    logger.info(f"  🎯 Prediction: {result['predicted_class']}")
                    logger.info(f"  📈 Confidence: {result['confidence']:.3f}")
                    logger.info(f"  ✅ Status: {'CORRECT' if is_correct else '❌ INCORRECT'}")
                    logger.info(f"  📊 Top 3 probabilities:")
                    sorted_probs = sorted(result['probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                    for class_name, prob in sorted_probs:
                        logger.info(f"    {class_name}: {prob:.3f}")
                    logger.info("")
                
                # Save individual prediction results
                with open(f"{inference_results_dir}/individual_predictions.json", 'w') as f:
                    json.dump(prediction_results, f, indent=2, default=str)
                logger.info(f"💾 Saved individual predictions to: {inference_results_dir}/individual_predictions.json")
                
                # Test batch prediction
                logger.info("🔄 Testing batch predictions:")
                batch_results = inference.predict_batch(test_texts, return_proba=True)
                correct_predictions = 0
                batch_prediction_results = []
                
                for i, (result, true_label) in enumerate(zip(batch_results, test_labels), 1):
                    is_correct = result['predicted_class'] == true_label
                    if is_correct:
                        correct_predictions += 1
                    
                    batch_result = {
                        'sample_id': i,
                        'true_label': true_label,
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'is_correct': is_correct
                    }
                    batch_prediction_results.append(batch_result)
                    
                    logger.info(f"🔄 Batch {i}: {result['predicted_class']} (True: {true_label}) "
                              f"(conf: {result['confidence']:.3f}) {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                # Calculate accuracy
                accuracy = correct_predictions / len(test_texts)
                logger.info(f"\n📊 Batch Prediction Accuracy: {accuracy:.3f} ({correct_predictions}/{len(test_texts)})")
                
                # Save batch prediction results
                with open(f"{inference_results_dir}/batch_predictions.json", 'w') as f:
                    json.dump(batch_prediction_results, f, indent=2, default=str)
                logger.info(f"💾 Saved batch predictions to: {inference_results_dir}/batch_predictions.json")
                
                # Create summary report
                summary_report = {
                    'test_date': datetime.now().isoformat(),
                    'model_info': info,
                    'test_samples_count': len(test_texts),
                    'batch_accuracy': accuracy,
                    'correct_predictions': correct_predictions,
                    'total_predictions': len(test_texts),
                    'test_classes': list(set(test_labels)),
                    'individual_predictions_file': 'individual_predictions.json',
                    'batch_predictions_file': 'batch_predictions.json'
                }
                
                with open(f"{inference_results_dir}/inference_summary.json", 'w') as f:
                    json.dump(summary_report, f, indent=2, default=str)
                logger.info(f"💾 Saved inference summary to: {inference_results_dir}/inference_summary.json")
                
                # Create CSV summary
                summary_data = []
                for result in prediction_results:
                    summary_data.append({
                        'Sample_ID': result['sample_id'],
                        'True_Label': result['true_label'],
                        'Predicted_Class': result['predicted_class'],
                        'Confidence': result['confidence'],
                        'Is_Correct': result['is_correct']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(f"{inference_results_dir}/inference_results_summary.csv", index=False)
                logger.info(f"💾 Saved inference summary CSV to: {inference_results_dir}/inference_results_summary.csv")
                
                # Final summary
                logger.info("\n" + "="*80)
                logger.info("🎉 TF-IDF INFERENCE TEST COMPLETED SUCCESSFULLY!")
                logger.info("="*80)
                logger.info(f"📊 Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{len(test_texts)})")
                logger.info(f"🤖 Model Used: {info['model_name']}")
                logger.info(f"📁 Results saved to: {inference_results_dir}")
                logger.info("📄 Files created:")
                logger.info("  - model_info.json (model details)")
                logger.info("  - individual_predictions.json (detailed individual results)")
                logger.info("  - batch_predictions.json (batch prediction results)")
                logger.info("  - inference_summary.json (overall summary)")
                logger.info("  - inference_results_summary.csv (CSV summary)")
                
            else:
                logger.warning("⚠️ Model files not found. Train a model first using train/tf_idf.py")
        else:
            logger.warning("⚠️ No model files found. Train a model first using train/tf_idf.py")
    else:
        logger.warning("⚠️ Models directory not found. Train a model first using train/tf_idf.py")
    
    logger.info("\n🏁 Inference module test completed")
