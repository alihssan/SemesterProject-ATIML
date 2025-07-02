"""
Doc2Vec Classification Pipeline
"""

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
from datetime import datetime

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def create_results_folder():
    """Create results folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/doc2vec_classification_{timestamp}"
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

def train_doc2vec_classifiers(dataframe, vectors_array, test_size=0.2, random_state=42):
    """
    Train multiple classifiers using Doc2Vec vectors and class labels.

    Args:
        dataframe (pd.DataFrame): DataFrame containing class labels and other metadata
        vectors_array (np.ndarray): Numpy array containing Doc2Vec vectors with shape (n_samples, size)
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

    # === 4. Define Parameter Grids (Optimized for Speed) ===
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
        logger.info(f"Starting grid search for: {name}")
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

        logger.info(f"Completed: {name} in {elapsed_time:.2f} seconds")
        logger.info(f"Best Parameters: {grid.best_params_}")
        
        # Display comprehensive metrics
        logger.info(f"\n{name} - COMPREHENSIVE METRICS:")
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
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        print(f"\nConfusion Matrix for {name}:")
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
    
    logger.info(f"\nAll results saved to: {results_dir}")
    logger.info("Files created:")
    logger.info("  - confusion_matrix_*.png (confusion matrices for each model)")
    logger.info("  - metrics_comparison.png (comparison of all metrics)")
    logger.info("  - training_time_comparison.png (training time comparison)")
    logger.info("  - f1_score_breakdown.png (F1-score by averaging method)")
    logger.info("  - classification_results_summary.csv (summary table)")
    logger.info("  - detailed_results.json (detailed results)")
    
    return results

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("STARTING DOC2VEC CLASSIFICATION PIPELINE")
    logger.info("="*80)
    
    # === Load Newsgroups Dataset ===
    logger.info("\nSTEP 1: Loading Newsgroups dataset...")
    
    # Import required modules with correct paths
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    logger.info("Import paths configured")
    
    from dataset.newsgroups_dataset import NewsgroupsDataset
    from vectorization.doc2vec import create_doc2vec_vectors_from_newsgroups
    logger.info("Required modules imported successfully")
    
    # Initialize datasets
    logger.info("Initializing NewsgroupsDataset...")
    dataset_raw = NewsgroupsDataset(remove_headers=True, remove_footers=True, remove_quotes=True, preprocess=False)
    dataset_proc = NewsgroupsDataset(remove_headers=True, remove_footers=True, remove_quotes=True, preprocess=True, remove_empty=True, clip_long_docs=True, clip_percentile=95)
    
    logger.info("NewsgroupsDataset initialized successfully")
    
    # Get DataFrames
    logger.info("\nSTEP 2: Preparing DataFrames...")
    df_raw = dataset_raw.create_dataframe(split='train', raw=True)
    df_proc = dataset_proc.create_dataframe(split='train', raw=False)
    
    logger.info(f"Raw DataFrame shape: {df_raw.shape}")
    logger.info(f"Processed DataFrame shape: {df_proc.shape}")
    
    # === Generate Doc2Vec Vectors ===
    logger.info("\nSTEP 3: Generating Doc2Vec vectors...")
    
    # Create results directory for visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_results_dir = f"results/doc2vec_visualizations_{timestamp}"
    os.makedirs(vis_results_dir, exist_ok=True)
    logger.info(f"Created visualization results directory: {vis_results_dir}")
    
    # Define visualization settings (save to results folder)
    visualization_params = {
        "visualize": True, 
        "method": "3d", 
        "vector_size": 100, 
        "epochs": 30,
        "save_dir": vis_results_dir
    }
    
    logger.info(f"Visualization parameters: {visualization_params}")
    
    # === Raw Dataset ===
    logger.info("\nRaw Text Doc2Vec Embeddings (3D)")
    vectors_raw = create_doc2vec_vectors_from_newsgroups(
        dataset=dataset_raw, 
        split='train', 
        use_preprocessed=False, 
        **visualization_params
    )
    df_vectors_raw = pd.DataFrame(vectors_raw, columns=[f"vec_{i}" for i in range(vectors_raw.shape[1])])
    df_vectors_raw['class_name'] = df_raw['class_name'].values
    
    logger.info(f"Raw Doc2Vec vector shape: {vectors_raw.shape}")
    
    # === Preprocessed Dataset ===
    logger.info("\nPreprocessed Text Doc2Vec Embeddings (3D)")
    vectors_proc = create_doc2vec_vectors_from_newsgroups(
        dataset=dataset_proc, 
        split='train', 
        use_preprocessed=True, 
        **visualization_params
    )
    df_vectors_proc = pd.DataFrame(vectors_proc, columns=[f"vec_{i}" for i in range(vectors_proc.shape[1])])
    df_vectors_proc['class_name'] = df_proc['class_name'].values
    
    logger.info(f"Processed Doc2Vec vector shape: {vectors_proc.shape}")
    
    # === Compare Raw vs Processed Performance ===
    logger.info("\nSTEP 4: Comparing Raw vs Processed Doc2Vec Performance...")
    
    # Train classifiers on both raw and processed vectors
    logger.info("Training classifiers on Raw Doc2Vec vectors...")
    dataframe_raw = df_vectors_raw[["class_name"]]
    results_raw = train_doc2vec_classifiers(dataframe_raw, vectors_raw)
    
    logger.info("Training classifiers on Processed Doc2Vec vectors...")
    dataframe_proc = df_vectors_proc[["class_name"]]
    results_proc = train_doc2vec_classifiers(dataframe_proc, vectors_proc)
    
    # === Compare Results ===
    logger.info("\nSTEP 5: Comparing Raw vs Processed Results...")
    
    # Create comparison summary
    comparison_data = []
    for model_name in results_raw.keys():
        raw_f1 = results_raw[model_name]['f1_macro']
        proc_f1 = results_proc[model_name]['f1_macro']
        improvement = proc_f1 - raw_f1
        
        comparison_data.append({
            'Model': model_name,
            'Raw F1-Score': raw_f1,
            'Processed F1-Score': proc_f1,
            'Improvement': improvement,
            'Better': 'Processed' if improvement > 0 else 'Raw'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f"{vis_results_dir}/raw_vs_processed_comparison.csv", index=False)
    
    logger.info("Raw vs Processed Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(comparison_data))
    width = 0.35
    
    raw_scores = [row['Raw F1-Score'] for row in comparison_data]
    proc_scores = [row['Processed F1-Score'] for row in comparison_data]
    
    plt.bar(x - width/2, raw_scores, width, label='Raw Doc2Vec', color='#FF6B6B', alpha=0.8)
    plt.bar(x + width/2, proc_scores, width, label='Processed Doc2Vec', color='#4ECDC4', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('F1-Score (Macro)')
    plt.title('Raw vs Processed Doc2Vec Performance Comparison')
    plt.xticks(x, [row['Model'] for row in comparison_data], rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, (raw_score, proc_score) in enumerate(zip(raw_scores, proc_scores)):
        plt.text(i - width/2, raw_score + 0.01, f'{raw_score:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.text(i + width/2, proc_score + 0.01, f'{proc_score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{vis_results_dir}/raw_vs_processed_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plot saved to: {vis_results_dir}/raw_vs_processed_comparison.png")
    
    # === Use Best Performing Vectors for Final Analysis ===
    logger.info("\nSTEP 6: Final Analysis with Best Performing Vectors...")
    
    # Determine which performs better overall
    avg_raw_f1 = np.mean([results_raw[model]['f1_macro'] for model in results_raw.keys()])
    avg_proc_f1 = np.mean([results_proc[model]['f1_macro'] for model in results_proc.keys()])
    
    if avg_proc_f1 > avg_raw_f1:
        logger.info(f"Processed Doc2Vec performs better (avg F1: {avg_proc_f1:.4f} vs {avg_raw_f1:.4f})")
        vectors_array = vectors_proc
        dataframe = dataframe_proc
        results = results_proc
        vector_type = "Processed"
    else:
        logger.info(f"Raw Doc2Vec performs better (avg F1: {avg_raw_f1:.4f} vs {avg_proc_f1:.4f})")
        vectors_array = vectors_raw
        dataframe = dataframe_raw
        results = results_raw
        vector_type = "Raw"
    
    logger.info(f"Using {vector_type} Doc2Vec vectors for final analysis")
    
    logger.info(f"Final vectors shape: {vectors_array.shape}")
    logger.info(f"Dataframe shape: {dataframe.shape}")
    logger.info(f"Number of features: {vectors_array.shape[1]}")
    logger.info(f"Number of samples: {vectors_array.shape[0]}")
    
    # Display feature information
    logger.info("\nFeature Analysis:")
    logger.info(f"Feature range: {vectors_array.min():.4f} to {vectors_array.max():.4f}")
    logger.info(f"Mean feature value: {vectors_array.mean():.4f}")
    logger.info(f"Standard deviation: {vectors_array.std():.4f}")
    logger.info(f"Non-zero features: {np.count_nonzero(vectors_array)} out of {vectors_array.size}")
    logger.info(f"Sparsity: {1 - np.count_nonzero(vectors_array) / vectors_array.size:.4f}")
    
    # Check for any NaN or infinite values
    nan_count = np.isnan(vectors_array).sum()
    inf_count = np.isinf(vectors_array).sum()
    logger.info(f"NaN values: {nan_count}")
    logger.info(f"Infinite values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        logger.warning("Data contains NaN or infinite values - this may affect model performance")
    
    # === Display Final Results Summary ===
    logger.info("\nSTEP 7: Final Results Summary...")
    
    # Create summary table for the best performing vectors
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
    logger.info(f"Vector Type Used: {vector_type}")
    
    logger.info("\n" + "="*80)
    logger.info("DOC2VEC CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    logger.info(f"\nResults saved to: {vis_results_dir}")
    logger.info(f"Raw vs Processed comparison saved to: {vis_results_dir}")
    logger.info(f"Classification results saved to individual model folders") 