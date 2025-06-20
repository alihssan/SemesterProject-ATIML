# 20 Newsgroups Analysis Pipeline

This project implements a comprehensive pipeline for analyzing the 20 Newsgroups dataset with multiple document representations, classifiers, and explainable AI techniques.

## 🎯 Pipeline Overview

The pipeline implements the following workflow:

1. **Load and Preprocess 20 Newsgroups**
   - Load dataset from sklearn
   - Clean and normalize text
   - Split into train/test sets

2. **Generate 3 Document Representations**
   - **TF-IDF Vectors**: Traditional bag-of-words approach
   - **Doc2Vec Embeddings**: Neural document embeddings
   - **Gemini 3B Summary → Sentence-BERT**: LLM summaries converted to vectors

3. **Train Classifiers on 20NG**
   - **SVM**: Support Vector Machine
   - **MLP**: Multi-layer Perceptron
   - **Decision Trees**: Random Forest and XGBoost
   - Evaluate with Accuracy, Precision, Recall, F1

4. **Generate Gemini Reasoning (XAI on 20NG)**
   - Generate Gemini 3B reasoning for selected predictions
   - Analyze model predictions with LLM reasoning

## 📋 Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Ollama with Gemma 3B

```bash
# Install Ollama (if not already installed)
# Follow instructions at: https://ollama.ai/

# Pull Gemma 3B model
ollama pull gemma:3b

# Start Ollama service
ollama serve
```

### 3. Verify Installation

```bash
# Test if Ollama is running
curl http://localhost:11434/api/tags
```

## 🚀 Quick Start

### Option 1: Run with Default Settings

```bash
python run_pipeline.py
```

### Option 2: Quick Test Run

```bash
python run_pipeline.py --quick
```

### Option 3: Custom Configuration

```bash
python run_pipeline.py \
    --categories comp.graphics comp.os.ms-windows.misc \
    --max_samples 300 \
    --test_size 0.25 \
    --random_state 123
```

## 📁 Project Structure

```
SemesterProject-ATIML/
├── main_pipeline.py          # Main pipeline implementation
├── run_pipeline.py           # Command-line runner
├── requirements.txt          # Dependencies
├── PIPELINE_README.md        # This file
├── vectorization/            # Document representation modules
│   ├── tf_idf.py            # TF-IDF vectorization
│   ├── doc2vec.py           # Doc2Vec embeddings
│   └── llm_summarisation.py # Gemini summarization
├── classifiers/              # Classification modules
│   └── classifier.py        # Unified classifier class
├── results/                  # Pipeline outputs
│   └── pipeline_YYYYMMDD_HHMMSS/
│       ├── data_info.json
│       ├── classifier_results.json
│       ├── summaries.csv
│       ├── gemini_analysis.pkl
│       └── visualizations/
└── dataset/                  # Dataset exploration
```

## 🔧 Configuration Options

### Pipeline Parameters

- `categories`: List of newsgroup categories to use
- `max_samples_per_category`: Maximum samples per category (for faster execution)
- `test_size`: Proportion of data for testing (default: 0.2)
- `random_state`: Random seed for reproducibility

### Classifier Configurations

The pipeline automatically trains:
- **SVM**: RBF kernel, C=1.0
- **Neural Network**: 2 hidden layers (100, 50), ReLU activation
- **Random Forest**: 100 estimators, default parameters

### Vectorization Settings

- **TF-IDF**: 5000 features, bigrams, min_df=2, max_df=0.95
- **Doc2Vec**: 100 dimensions, window=5, epochs=20
- **Sentence-BERT**: all-MiniLM-L6-v2 model

## 📊 Output Files

### 1. Data Information
- `data_info.json`: Dataset statistics and category information

### 2. Classifier Results
- `classifier_results.json`: Summary of all classifier performances
- `detailed_results.pkl`: Detailed evaluation metrics

### 3. Document Representations
- `tfidf_features.pkl`: TF-IDF feature names
- `summaries.csv`: Gemini-generated summaries

### 4. Gemini Analysis
- `gemini_analysis.pkl`: Gemini reasoning for predictions

### 5. Visualizations
- `classifier_performance.png`: Performance comparison chart
- `feature_importance.png`: Top features from Random Forest
- `confusion_matrix.png`: Confusion matrix for best classifier

## 🔍 Understanding the Results

### Classifier Performance

The pipeline compares three classifiers across three representations:

1. **TF-IDF**: Traditional approach, most interpretable
2. **Doc2Vec**: Neural embeddings, captures semantic relationships
3. **Sentence-BERT**: LLM-enhanced approach, combines summarization with embeddings

### Gemini Analysis

For each analyzed prediction, you'll find:

1. **Gemini Reasoning**: LLM-generated reasoning for the prediction
2. **Prediction Confidence**: Model's confidence in the prediction
3. **Correctness**: Whether the prediction was correct
4. **Text Analysis**: LLM's analysis of why the text belongs to the predicted category

## 🛠️ Customization

### Adding New Classifiers

```python
# In main_pipeline.py, modify classifier_configs
classifier_configs = {
    'svm': {'classifier_type': 'svm', 'C': 1.0, 'kernel': 'rbf'},
    'neural_network': {'classifier_type': 'neural_network', 'hidden_layer_sizes': (100, 50)},
    'random_forest': {'classifier_type': 'random_forest', 'n_estimators': 100},
    'xgboost': {'classifier_type': 'xgboost', 'n_estimators': 100}  # Add new classifier
}
```

### Modifying Vectorization

```python
# In the pipeline methods, adjust parameters
self.tfidf_vectors, feature_names = create_tfidf_vectors(
    all_texts,
    max_features=10000,  # Increase features
    ngram_range=(1, 3),  # Use trigrams
    return_feature_names=True
)
```

### Changing Gemini Analysis Settings

```python
# In generate_gemini_reasoning method
sample_indices = np.random.choice(len(self.X_test), min(10, len(self.X_test)), replace=False)  # More samples
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Error: Failed to connect to Ollama
   Solution: Ensure Ollama is running: ollama serve
   ```

2. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce max_samples_per_category or use --quick flag
   ```

3. **Missing Dependencies**
   ```
   Error: ModuleNotFoundError
   Solution: Install requirements: pip install -r requirements.txt
   ```

4. **Slow Execution**
   ```
   Solution: Use --quick flag or reduce max_samples_per_category
   ```

### Performance Tips

- Use `--quick` flag for testing
- Reduce `max_samples_per_category` for faster execution
- Use fewer categories for initial testing
- Ensure sufficient RAM (8GB+ recommended)

## 📈 Expected Results

### Typical Performance (4 categories, 200 samples each)

| Representation | SVM | Neural Network | Random Forest |
|----------------|-----|----------------|---------------|
| TF-IDF         | ~85% | ~83% | ~87% |
| Doc2Vec        | ~82% | ~80% | ~84% |
| Sentence-BERT  | ~86% | ~84% | ~88% |

*Note: Results may vary based on data and hardware*

## 🤝 Contributing

To extend the pipeline:

1. Add new vectorization methods in `vectorization/`
2. Add new classifiers in `classifiers/`
3. Modify `main_pipeline.py` to include new components
4. Update `requirements.txt` with new dependencies

## 📄 License

This project is part of the ATIML Semester Project.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in the results directory
3. Ensure all dependencies are properly installed
4. Verify Ollama is running with Gemma 3B model 