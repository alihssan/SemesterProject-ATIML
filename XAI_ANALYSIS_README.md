# XAI Analysis Pipeline

This pipeline provides comprehensive explainability analysis for text classification models using LIME (Local Interpretable Model-agnostic Explanations) and LLM reasoning.

## 🎯 Overview

The XAI Analysis Pipeline combines:
- **LIME**: For local feature importance explanations
- **LLM Reasoning**: For natural language explanations using Ollama/Gemma
- **Agreement Analysis**: To measure consistency between LIME and LLM explanations

## 📊 Supported Datasets

1. **20 Newsgroups Dataset**: 20 different newsgroup categories
2. **AG News Dataset**: 4 news categories (World, Sports, Business, Sci/Tech)

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Ollama** (for LLM explanations):
   ```bash
   # Install Ollama
   # Follow instructions at: https://ollama.ai/
   
   # Pull Gemma model
   ollama pull gemma:3b
   
   # Start Ollama service
   ollama serve
   ```

### Running the Pipeline

#### Option 1: Analyze Both Datasets
```bash
python run_xai_analysis.py --dataset both --samples 15
```

#### Option 2: Analyze Specific Dataset
```bash
# 20 Newsgroups only
python run_xai_analysis.py --dataset newsgroups --samples 20

# AG News only
python run_xai_analysis.py --dataset ag_news --samples 20
```

#### Option 3: Custom Configuration
```bash
python run_xai_analysis.py \
    --dataset both \
    --samples 25 \
    --random_state 123 \
    --output_dir results/my_analysis
```

## 📁 Output Structure

For each dataset, the pipeline creates a results directory with:

```
results/xai_analysis_{dataset}_{timestamp}/
├── explanations.json              # Detailed explanations for each sample
├── summary_statistics.json        # Overall statistics
├── classification_report.json     # Model performance metrics
├── xai_analysis_results.pkl       # Pickle file for further analysis
├── detailed_explanations.txt      # Human-readable explanation samples
├── agreement_distribution.png     # Distribution of agreement scores
├── lime_feature_importance.png    # Top LIME features
└── confidence_vs_agreement.png    # Confidence vs agreement correlation
```

## 🔍 Understanding the Results

### 1. LIME Explanations
- **Feature Importance**: Which words/phrases contributed most to the classification
- **Weights**: Positive/negative contribution of each feature
- **Local Interpretability**: Explanations specific to each prediction

### 2. LLM Explanations
- **Natural Language**: Human-readable reasoning for classifications
- **Contextual Analysis**: Understanding of why text belongs to a category
- **LIME-Enhanced**: LLM explanations that incorporate LIME insights

### 3. Agreement Analysis
- **Agreement Score**: Measures how well LLM explanations align with LIME features
- **Confidence Correlation**: Relationship between model confidence and explanation agreement
- **Feature Overlap**: Which LIME features are mentioned in LLM explanations

## 📈 Key Metrics

### Agreement Score
- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - 0.0-0.3: Low agreement (LLM doesn't mention LIME features)
  - 0.3-0.7: Medium agreement (partial overlap)
  - 0.7-1.0: High agreement (LLM incorporates LIME insights well)

### Classification Performance
- **Accuracy**: Overall classification accuracy
- **Confidence**: Model's confidence in predictions
- **Per-class Performance**: Detailed metrics for each category

## 🛠️ Customization

### Modifying LIME Parameters
```python
# In xai_analysis_pipeline.py
def _get_lime_explanation(self, text: str, num_features: int = 10) -> Dict:
    # Change num_features for more/less detailed explanations
    exp = self.lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=20,  # More features
        num_samples=200   # More samples for better accuracy
    )
```

### Adjusting LLM Prompts
```python
# In _get_llm_explanation method
prompt = f"""
Analyze the following text and explain why it was classified as "{predicted_label}" 
(true label: "{true_label}") in the context of {self.dataset_type} classification.

Text: {text[:1000]}...

Please provide a natural language explanation focusing on:
1. Key words or phrases that indicate the topic
2. Linguistic patterns that suggest the category
3. Why this classification makes sense (or doesn't)
4. Additional context or insights

Keep your explanation concise but informative.
"""
```

### Changing Sample Selection
```python
# In analyze_samples method
# Select diverse samples based on different criteria
sample_indices = np.random.choice(len(self.X_test), self.num_samples, replace=False)

# Or select samples with specific characteristics
# high_confidence_indices = np.where(confidences > 0.8)[0]
# sample_indices = np.random.choice(high_confidence_indices, self.num_samples, replace=False)
```

## 🔧 Advanced Usage

### Using the Pipeline Programmatically
```python
from xai_analysis_pipeline import XAIAnalysisPipeline

# Create pipeline for 20 Newsgroups
pipeline = XAIAnalysisPipeline(
    dataset_type='newsgroups',
    num_samples=20,
    random_state=42
)

# Run analysis
pipeline.run_complete_analysis()

# Access results
for explanation in pipeline.explanations:
    print(f"Sample: {explanation.document_id}")
    print(f"LIME features: {explanation.lime_explanation['features'][:3]}")
    print(f"Agreement score: {explanation.agreement_score:.3f}")
    print("---")
```

### Comparing Datasets
```python
# Run analysis for both datasets
newsgroups_pipeline = XAIAnalysisPipeline(dataset_type='newsgroups')
ag_news_pipeline = XAIAnalysisPipeline(dataset_type='ag_news')

# Compare agreement scores
newsgroups_agreement = [exp.agreement_score for exp in newsgroups_pipeline.explanations]
ag_news_agreement = [exp.agreement_score for exp in ag_news_pipeline.explanations]

print(f"20 Newsgroups mean agreement: {np.mean(newsgroups_agreement):.3f}")
print(f"AG News mean agreement: {np.mean(ag_news_agreement):.3f}")
```

## 🐛 Troubleshooting

### Common Issues

1. **LIME Import Error**:
   ```
   Error: LIME not available
   Solution: pip install lime
   ```

2. **Ollama Connection Error**:
   ```
   Error: Ollama LLM not available
   Solution: Ensure Ollama is running: ollama serve
   ```

3. **Memory Issues**:
   ```
   Error: Out of memory
   Solution: Reduce num_samples or use smaller dataset
   ```

4. **Dataset Loading Error**:
   ```
   Error: Dataset not found
   Solution: Ensure dataset files are in the correct location
   ```

### Performance Tips

1. **Reduce Sample Size**: Use fewer samples for faster analysis
2. **Limit Text Length**: Truncate long texts to reduce processing time
3. **Use Smaller Models**: Consider using smaller LLM models for faster inference
4. **Parallel Processing**: Process multiple samples in parallel (future enhancement)

## 📚 References

- **LIME Paper**: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?"
- **Explainable AI**: Molnar, C. (2020). "Interpretable Machine Learning"
- **Text Classification**: Jurafsky, D., & Martin, J. H. (2020). "Speech and Language Processing"

## 🤝 Contributing

To extend the pipeline:

1. **Add New Datasets**: Implement dataset loading in `load_dataset()`
2. **New Explanation Methods**: Add methods for additional XAI techniques
3. **Enhanced Visualizations**: Create new plotting functions
4. **Performance Metrics**: Add additional agreement/consistency measures

## 📄 License

This project is part of the ATIML Semester Project. See the main README for license information. 