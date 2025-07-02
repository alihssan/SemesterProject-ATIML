import pandas as pd
import numpy as np
import logging
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import ast
from datetime import datetime

warnings.filterwarnings("ignore")

# === Setup Logging ===
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_results_folder():
    """Create results folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/tfidf_classification_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Created results directory: {results_dir}")
    return results_dir

def plot_confusion_matrix(cm, class_names, model_name, results_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    filename = f"{results_dir}/confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix: {filename}")

def plot_metrics_comparison(results, results_dir):
    """Plot comparison of all metrics across models."""
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results[model][metric] for model in results.keys()]
        models = list(results.keys())
        
        bars = axes[i].bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[i].set_title(f'{metric_name} Comparison')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    filename = f"{results_dir}/metrics_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics comparison: {filename}")

def plot_training_time_comparison(results, results_dir):
    """Plot training time comparison."""
    models = list(results.keys())
    times = [results[model]['training_time'] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    plt.title('Training Time Comparison')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"{results_dir}/training_time_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training time comparison: {filename}")

def plot_f1_score_breakdown(results, results_dir):
    """Plot F1-score breakdown by averaging method."""
    models = list(results.keys())
    f1_macro = [results[model]['f1_macro'] for model in models]
    f1_micro = [results[model]['f1_micro'] for model in models]
    f1_weighted = [results[model]['f1_weighted'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, f1_macro, width, label='F1 (Macro)', color='#FF6B6B')
    plt.bar(x, f1_micro, width, label='F1 (Micro)', color='#4ECDC4')
    plt.bar(x + width, f1_weighted, width, label='F1 (Weighted)', color='#45B7D1')
    
    plt.title('F1-Score Comparison by Averaging Method')
    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    filename = f"{results_dir}/f1_score_breakdown.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved F1-score breakdown: {filename}")

def save_detailed_results(results, results_dir):
    """Save detailed results to CSV and JSON files."""
    # Create summary DataFrame
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision (Macro)': result['precision_macro'],
            'Recall (Macro)': result['recall_macro'],
            'F1-Score (Macro)': result['f1_macro'],
            'Precision (Micro)': result['precision_micro'],
            'Recall (Micro)': result['recall_micro'],
            'F1-Score (Micro)': result['f1_micro'],
            'Precision (Weighted)': result['precision_weighted'],
            'Recall (Weighted)': result['recall_weighted'],
            'F1-Score (Weighted)': result['f1_weighted'],
            'Training Time (s)': result['training_time']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_filename = f"{results_dir}/classification_results_summary.csv"
    summary_df.to_csv(csv_filename, index=False)
    logger.info(f"Saved summary CSV: {csv_filename}")
    
    # Save detailed results to JSON
    import json
    detailed_results = {}
    for model_name, result in results.items():
        detailed_results[model_name] = {
            'accuracy': result['accuracy'],
            'precision_macro': result['precision_macro'],
            'recall_macro': result['recall_macro'],
            'f1_macro': result['f1_macro'],
            'precision_micro': result['precision_micro'],
            'recall_micro': result['recall_micro'],
            'f1_micro': result['f1_micro'],
            'precision_weighted': result['precision_weighted'],
            'recall_weighted': result['recall_weighted'],
            'f1_weighted': result['f1_weighted'],
            'training_time': result['training_time'],
            'best_params': result['best_params'],
            'confusion_matrix': result['confusion_matrix'].tolist()
        }
    
    json_filename = f"{results_dir}/detailed_results.json"
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    logger.info(f"Saved detailed JSON: {json_filename}")
    
    return summary_df

def train_tfidf_classifiers(dataframe, vectors_array, test_size=0.2, random_state=42):
    """
    Train multiple classifiers using TF-IDF vectors and class labels.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing class labels and other metadata
        vectors_array (np.ndarray): Numpy array containing TF-IDF vectors with shape (n_samples, size)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing trained models and their results
    """
    
    # Create results directory
    results_dir = create_results_folder()
    
    # === 1. Process and Combine Data ===
    logger.info("Processing and combining vectors with class information...")
    
    # Ensure vectors array and dataframe have same number of rows
    if len(vectors_array) != len(dataframe):
        raise ValueError(f"Vectors array length ({len(vectors_array)}) must match dataframe length ({len(dataframe)})")
    
    # Create feature names for the vectors
    n_features = vectors_array.shape[1]
    feature_names = [f"vec_{i}" for i in range(n_features)]
    
    # Create vectors dataframe
    vectors_df = pd.DataFrame(vectors_array, columns=feature_names)
    
    # Combine vectors with class information
    combined_df = pd.concat([dataframe.reset_index(drop=True), vectors_df.reset_index(drop=True)], axis=1)
    
    logger.info(f"Combined data shape: {combined_df.shape}")
    logger.info(f"Number of features: {n_features}")
    
    # === 2. Extract Features and Encode Target ===
    logger.info("Extracting features and encoding target labels...")
    X = combined_df[feature_names]  # Use all vector features
    y = combined_df["class_name"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info("Features and labels prepared.")

    # === 3. Train/Test Split ===
    logger.info("Splitting data into train and test sets...")
    X_train_vec, X_test_vec, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state)
    logger.info(f"Train set size: {X_train_vec.shape[0]} | Test set size: {X_test_vec.shape[0]}")

    # === 4. Define Parameter Grids (Optimized for Speed + Non-linear SVM) ===
    param_grids = {
        "SVC (RBF)": {
            "kernel": ['rbf'],              # Non-linear kernel
            "C": [0.1, 1, 10],              # Regularization
            "gamma": ['scale', 'auto']     # Kernel coefficient
        },
        "MLPClassifier": {
            "hidden_layer_sizes": [(100,), (50, 50)],  # Shallow vs deep MLP
            "activation": ['relu'],                    # Efficient activation
            "max_iter": [200]                          # Fast convergence
        },
        "SGDClassifier": {
            "loss": ['log_loss'],        # Logistic regression
            "alpha": [1e-4, 1e-3],       # Regularization strength
            "max_iter": [500]
        },
        "RandomForest": {
            "n_estimators": [100],       # Fixed number of trees
            "max_depth": [10, None]      # Limit vs grow fully
        },
        "XGBoost": {
            "n_estimators": [100],
            "max_depth": [3, 6],         # Shallow and moderate depth
            "learning_rate": [0.1],      
            "use_label_encoder": [False],
            "eval_metric": ['mlogloss']
        }
    }


    # === 5. Define Models ===
    base_models = {
        "SVC (RBF)": SVC(),
        "MLPClassifier": MLPClassifier(),
        "SGDClassifier": SGDClassifier(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier()
    }

    # === 6. Train, Tune, and Evaluate Each Model ===
    results = {}
    
    for name, model in base_models.items():
        logger.info(f"🔍 Starting grid search for: {name}")
        start_time = time.time()

        grid = GridSearchCV(model, param_grids[name], cv=2, scoring='accuracy', n_jobs=-1)  # Reduced CV folds
        grid.fit(X_train_vec, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_vec)

        # Calculate comprehensive metrics
        acc = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Micro averages
        precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
        
        # Weighted averages
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        elapsed_time = time.time() - start_time

        logger.info(f"✅ Completed: {name} in {elapsed_time:.2f} seconds")
        logger.info(f"Best Parameters: {grid.best_params_}")
        
        # Display comprehensive metrics
        logger.info(f"\n📊 {name} - COMPREHENSIVE METRICS:")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Precision (Macro): {precision_macro:.4f}")
        logger.info(f"  Recall (Macro): {recall_macro:.4f}")
        logger.info(f"  F1-Score (Macro): {f1_macro:.4f}")
        logger.info(f"  Precision (Micro): {precision_micro:.4f}")
        logger.info(f"  Recall (Micro): {recall_micro:.4f}")
        logger.info(f"  F1-Score (Micro): {f1_micro:.4f}")
        logger.info(f"  Precision (Weighted): {precision_weighted:.4f}")
        logger.info(f"  Recall (Weighted): {recall_weighted:.4f}")
        logger.info(f"  F1-Score (Weighted): {f1_weighted:.4f}")
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        print(f"\n📊 Confusion Matrix for {name}:")
        print(cm)
        
        # Store results
        results[name] = {
            'model': best_model,
            'accuracy': acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'best_params': grid.best_params_,
            'training_time': elapsed_time,
            'predictions': y_pred,
            'true_labels': y_test,
            'label_encoder': label_encoder,
            'feature_names': feature_names,
            'confusion_matrix': cm
        }
    
    # === 7. Generate and Save Visualizations ===
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS AND SAVING RESULTS")
    logger.info("="*60)
    
    # Plot confusion matrices
    for name, result in results.items():
        plot_confusion_matrix(result['confusion_matrix'], 
                            label_encoder.classes_, name, results_dir)
    
    # Plot metrics comparison
    plot_metrics_comparison(results, results_dir)
    
    # Plot training time comparison
    plot_training_time_comparison(results, results_dir)
    
    # Plot F1-score breakdown
    plot_f1_score_breakdown(results, results_dir)
    
    # Save detailed results
    summary_df = save_detailed_results(results, results_dir)
    
    # === 8. Save Best Model for Inference ===
    logger.info("\n" + "="*60)
    logger.info("SAVING BEST MODEL FOR INFERENCE")
    logger.info("="*60)
    
    # Find best model based on F1-score
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_macro'])
    best_result = results[best_model_name]
    
    logger.info(f"🏆 Best model: {best_model_name} (F1: {best_result['f1_macro']:.4f})")
    
    # Save best model for inference
    from inference.tf_idf import save_model_for_inference
    
    # Create TF-IDF vectorizer from the training data
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Get the processed texts
    processed_texts = [' '.join(tokens) for tokens in df_proc['data']]
    
    # Use the same TF-IDF vectorizer that was used for training
    # (tfidf_vectorizer is already defined above)
    
    # Save model for inference
    model_path, tfidf_path, encoder_path = save_model_for_inference(
        model=best_result['model'],
        tfidf_vectorizer=tfidf_vectorizer,
        label_encoder=label_encoder,
        model_name="tfidf_classifier",
        save_dir="models"
    )
    
    logger.info(f"\n🎉 All results saved to: {results_dir}")
    logger.info("Files created:")
    logger.info("  - confusion_matrix_*.png (confusion matrices for each model)")
    logger.info("  - metrics_comparison.png (comparison of all metrics)")
    logger.info("  - training_time_comparison.png (training time comparison)")
    logger.info("  - f1_score_breakdown.png (F1-score by averaging method)")
    logger.info("  - classification_results_summary.csv (summary table)")
    logger.info("  - detailed_results.json (detailed results)")
    logger.info("  - models/tfidf_classifier_*.pkl (inference models)")
    
    return results

# === Main execution (for backward compatibility) ===
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🚀 STARTING TF-IDF CLASSIFICATION PIPELINE")
    logger.info("="*80)
    
    # === Load Newsgroups Dataset ===
    logger.info("\n📚 STEP 1: Loading Newsgroups dataset...")
    
    # Import required modules with correct paths
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    logger.info("✅ Import paths configured")
    
    from dataset.newsgroups_dataset import NewsgroupsDataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    logger.info("✅ Required modules imported successfully")
    
    # Dataset initialization with NO preprocessing (raw text only)
    logger.info("🔧 Initializing NewsgroupsDataset with NO preprocessing (raw text)...")
    dataset = NewsgroupsDataset(
        remove_headers=True, 
        remove_footers=True, 
        remove_quotes=True, 
        preprocess=False,  # No tokenization/preprocessing
        remove_empty=True,
        clip_long_docs=False  # No clipping
    )
    logger.info("✅ NewsgroupsDataset initialized successfully")
    
    # Get DataFrame with raw text (training only)
    logger.info("\n🔄 STEP 2: Preparing DataFrame with raw text...")
    df_raw = dataset.create_dataframe(split='train', raw=True)
    
    logger.info(f"✅ DataFrame created. Shape: {df_raw.shape}")
    
    # === Generate TF-IDF Vectors from Raw Text ===
    logger.info("\n🔤 STEP 3: Generating TF-IDF vectors from raw text...")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/tfidf_raw_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Created results directory: {results_dir}")
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Fit and transform the raw texts
    raw_texts = df_raw['text'].tolist()
    X_tfidf = tfidf_vectorizer.fit_transform(raw_texts)
    
    logger.info(f"✅ TF-IDF vectors created. Shape: {X_tfidf.shape}")
    
    # Create DataFrame with TF-IDF vectors and labels
    tfidf_df = pd.DataFrame({
        "tfidf_vector": list(X_tfidf.toarray()),
        "class_name": df_raw['class_name']
    })
    
    # === Train Classifiers ===
    logger.info("\n🎯 STEP 4: Training classifiers...")
    
    # Prepare data for training
    dataframe = tfidf_df[["class_name"]]
    vectors_array = X_tfidf.toarray()
    
    # Train classifiers (this function handles the train/test split internally)
    results = train_tfidf_classifiers(dataframe, vectors_array, test_size=0.2, random_state=42)
    
    # === Save TF-IDF Data for Later Use ===
    logger.info("\n💾 STEP 5: Saving TF-IDF data...")
    
    # Save the raw text data with TF-IDF vectors
    tfidf_data_df = pd.DataFrame({
        'original_text': raw_texts,
        'tfidf_vector': list(X_tfidf.toarray()),
        'label': df_raw['label'],
        'class_name': df_raw['class_name']
    })
    
    # Split into train/test for consistency
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        tfidf_data_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=tfidf_data_df['label']
    )
    
    # Save to CSV files
    full_csv_filename = f"{results_dir}/tfidf_full_{timestamp}.csv"
    train_csv_filename = f"{results_dir}/tfidf_train_{timestamp}.csv"
    test_csv_filename = f"{results_dir}/tfidf_test_{timestamp}.csv"
    
    tfidf_data_df.to_csv(full_csv_filename, index=False)
    train_df.to_csv(train_csv_filename, index=False)
    test_df.to_csv(test_csv_filename, index=False)
    
    logger.info(f"💾 Saved full dataset to: {full_csv_filename}")
    logger.info(f"💾 Saved train split to: {train_csv_filename}")
    logger.info(f"💾 Saved test split to: {test_csv_filename}")
    
    # === Display Final Results Summary ===
    logger.info("\nSTEP 6: Final Results Summary...")
    
    # Create summary table
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision (Macro)': metrics['precision_macro'],
            'Recall (Macro)': metrics['recall_macro'],
            'F1-Score (Macro)': metrics['f1_macro'],
            'Precision (Micro)': metrics['precision_micro'],
            'Recall (Micro)': metrics['recall_micro'],
            'F1-Score (Micro)': metrics['f1_micro'],
            'Training Time (s)': metrics['training_time']
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info("\nFinal Classification Results Summary:")
    logger.info(summary_df.to_string(index=False))
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['f1_macro'])
    best_f1 = results[best_model]['f1_macro']
    best_accuracy = results[best_model]['accuracy']
    
    logger.info(f"\nBEST MODEL: {best_model}")
    logger.info(f"Best F1-Score (Macro): {best_f1:.4f}")
    logger.info(f"Best Accuracy: {best_accuracy:.4f}")
    logger.info(f"Training Time: {results[best_model]['training_time']:.2f} seconds")
    logger.info(f"Vector Type Used: Raw Text TF-IDF")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 TF-IDF CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"TF-IDF data saved to: {results_dir}")
    logger.info(f"Classification results saved to individual model folders")
