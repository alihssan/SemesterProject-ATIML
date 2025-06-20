# AG News Pipeline with Zero-Shot and Few-Shot Learning

This pipeline implements modern text classification approaches for the AG News dataset, comparing zero-shot, few-shot, and traditional supervised learning methods.

## 🎯 Features

- **Zero-Shot Classification**: Using pre-trained language models (BART-large-MNLI)
- **Few-Shot Learning**: With 1, 3, 5, and 10 shots per class
- **Supervised Learning**: Traditional TF-IDF + Logistic Regression baseline
- **Comprehensive Evaluation**: Accuracy, F1-score, precision, recall
- **Visualization**: Comparison charts and learning curves
- **Flexible Configuration**: Easy to customize and extend

## 📊 Dataset

The AG News dataset contains news articles from 4 categories:
- **World** (0): International news and events
- **Sports** (1): Sports-related articles
- **Business** (2): Business and financial news
- **Sci/Tech** (3): Science and technology articles

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default settings
python run_ag_news_pipeline.py

# Run with custom parameters
python run_ag_news_pipeline.py --max-samples 1000 --few-shot-shots 1 5 10
```

### Advanced Usage

```bash
# Skip specific components
python run_ag_news_pipeline.py --skip-zero-shot
python run_ag_news_pipeline.py --skip-few-shot
python run_ag_news_pipeline.py --skip-supervised

# Custom configuration
python run_ag_news_pipeline.py --max-samples 200 --few-shot-shots 1 3 5 --random-state 123
```

## 📋 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-samples` | int | 500 | Maximum samples per class |
| `--few-shot-shots` | list | [1,3,5,10] | Number of shots for few-shot learning |
| `--random-state` | int | 42 | Random seed for reproducibility |
| `--config-mode` | str | 'ag_news' | Configuration mode |
| `--skip-zero-shot` | flag | False | Skip zero-shot classification |
| `--skip-few-shot` | flag | False | Skip few-shot classification |
| `--skip-supervised` | flag | False | Skip supervised classification |

## 🔧 Configuration

The pipeline uses a configuration file (`config/ag_news_config.py`) with the following sections:

### Zero-Shot Configuration
```python
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
    'device': 'auto'
}
```

### Few-Shot Configuration
```python
'few_shot': {
    'enabled': True,
    'embedding_model': 'all-MiniLM-L6-v2',
    'classifier': 'logistic_regression',
    'classifier_params': {
        'random_state': 42,
        'max_iter': 1000,
        'C': 1.0
    }
}
```

## 📈 Output

The pipeline generates several outputs in the `results/` directory:

### Files Generated
- `ag_news_comparison_chart.png` - Main comparison visualization
- `ag_news_comparison_data.csv` - Raw data for the chart
- `zero_shot_results.pkl` - Zero-shot classification results
- `few_shot_results.pkl` - Few-shot classification results
- `supervised_results.pkl` - Supervised classification results
- `dataset_info.json` - Dataset statistics and information
- `ag_news_pipeline.log` - Detailed execution log

### Visualization
The main chart shows:
- **Blue bars**: Accuracy scores
- **Red bars**: F1-score values
- **Methods compared**: Zero-Shot, 1-Shot, 3-Shot, 5-Shot, 10-Shot, Supervised
- **Value labels**: Exact scores on each bar
- **Baseline reference**: Gray dashed line at 0.5

## 🔬 Methods Explained

### Zero-Shot Classification
- Uses pre-trained language models (BART-large-MNLI)
- No training examples needed
- Classifies text based on natural language understanding
- Good for quick prototyping and low-resource scenarios

### Few-Shot Learning
- Uses sentence embeddings (all-MiniLM-L6-v2)
- Trains a simple classifier on few examples per class
- Balances between zero-shot and supervised learning
- Useful when limited labeled data is available

### Supervised Learning
- Traditional TF-IDF + Logistic Regression
- Uses full training dataset
- Serves as performance baseline
- Most accurate but requires significant labeled data

## 📊 Expected Results

Typical performance on AG News dataset:

| Method | Accuracy | F1-Score | Training Examples |
|--------|----------|----------|-------------------|
| Zero-Shot | ~0.65-0.75 | ~0.65-0.75 | 0 |
| 1-Shot | ~0.70-0.80 | ~0.70-0.80 | 4 |
| 3-Shot | ~0.75-0.85 | ~0.75-0.85 | 12 |
| 5-Shot | ~0.80-0.90 | ~0.80-0.90 | 20 |
| 10-Shot | ~0.85-0.92 | ~0.85-0.92 | 40 |
| Supervised | ~0.90-0.95 | ~0.90-0.95 | 1000+ |

*Note: Results may vary based on data sampling and model initialization*

## 🛠️ Requirements

### Core Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Optional Dependencies
```bash
# For zero-shot classification
pip install transformers torch

# For sentence embeddings
pip install sentence-transformers

# For dataset loading
pip install datasets
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   export CUDA_VISIBLE_DEVICES=""
   python run_ag_news_pipeline.py
   ```

2. **Slow Zero-Shot Processing**
   ```bash
   # Reduce batch size in config
   'batch_size': 16  # or 8
   ```

3. **Dataset Loading Issues**
   ```bash
   # Check internet connection
   # Try with smaller dataset
   python run_ag_news_pipeline.py --max-samples 100
   ```

### Performance Tips

- Use GPU for faster zero-shot classification
- Reduce `max_samples` for faster processing
- Skip components you don't need with `--skip-*` flags
- Use smaller few-shot configurations for quick testing

## 🔄 Extending the Pipeline

### Adding New Zero-Shot Models
```python
# In ag_news_pipeline.py
def setup_zero_shot_classifier(self):
    # Change model name
    model_name = "your-preferred-model"
    self.zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
```

### Adding New Few-Shot Classifiers
```python
# In run_few_shot_classification method
from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=self.random_state)
```

### Custom Prompts
```python
# In config/ag_news_config.py
'prompts': [
    "Your custom prompt here:",
    "Another custom prompt:"
]
```

## 📚 References

- [AG News Dataset](https://huggingface.co/datasets/ag_news)
- [BART-large-MNLI](https://huggingface.co/facebook/bart-large-mnli)
- [Sentence Transformers](https://www.sbert.net/)
- [Few-Shot Learning](https://arxiv.org/abs/2005.14165)

## 🤝 Contributing

Feel free to contribute by:
- Adding new zero-shot models
- Implementing different few-shot approaches
- Improving visualizations
- Adding more evaluation metrics
- Optimizing performance

## 📄 License

This project is part of the ATIML Semester Project. 