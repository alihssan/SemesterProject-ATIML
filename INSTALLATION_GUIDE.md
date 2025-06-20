# Installation Guide

This guide will help you install the dependencies for the 20 Newsgroups Pipeline without encountering the scipy installation issue.

## 🚨 The Scipy Issue

The main issue is that newer versions of scikit-learn require scipy, and scipy requires OpenBLAS which can be difficult to install on some systems. We've solved this by using scikit-learn 1.2.2 which doesn't require scipy.

## 📋 Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- Virtual environment (recommended)

### 2. Ollama Setup (for Gemini features)
```bash
# Install Ollama
# Follow instructions at: https://ollama.ai/

# Pull Gemma 3B model
ollama pull gemma:3b

# Start Ollama service
ollama serve
```

## 🚀 Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install core dependencies
pip install -r requirements.txt
```

### Method 2: Step-by-Step Install

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install core packages first
pip install numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 requests>=2.31.0 tqdm>=4.65.0

# 3. Install scikit-learn with specific version (no scipy dependency)
pip install scikit-learn==1.2.2

# 4. Install gensim
pip install gensim>=4.3.0
```

### Method 3: Using Conda (Alternative)

```bash
# 1. Create conda environment
conda create -n newsgroups python=3.9
conda activate newsgroups

# 2. Install packages via conda
conda install numpy pandas scikit-learn=1.2.2 matplotlib seaborn requests tqdm

# 3. Install gensim via pip
pip install gensim>=4.3.0
```

## 🔧 Optional Dependencies

After installing the core dependencies, you can optionally install additional packages:

### XAI Packages (for explainable AI)
```bash
pip install lime shap
```

### Sentence Transformers (for advanced embeddings)
```bash
pip install sentence-transformers transformers torch
```

### Additional ML Packages
```bash
pip install xgboost plotly wordcloud
```

## ✅ Verification

Test your installation:

```bash
# Test core functionality
python -c "
import numpy
import pandas
import sklearn
import gensim
import matplotlib
import seaborn
print('✓ All core packages installed successfully!')
"

# Test pipeline
python run_pipeline.py --quick
```

## 🐛 Troubleshooting

### Issue: Scipy Installation Fails
**Solution**: Use scikit-learn 1.2.2 instead of newer versions
```bash
pip uninstall scikit-learn
pip install scikit-learn==1.2.2
```

### Issue: OpenBLAS Not Found
**Solution**: Use conda or install scikit-learn 1.2.2
```bash
conda install scikit-learn=1.2.2
```

### Issue: Ollama Connection Error
**Solution**: Ensure Ollama is running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### Issue: Memory Issues
**Solution**: Use smaller dataset
```bash
python run_pipeline.py --quick --max_samples 50
```

### Issue: Missing Packages
**Solution**: Install missing packages individually
```bash
pip install package_name
```

## 📦 Package Versions

### Core Packages (Required)
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn == 1.2.2 (specific version to avoid scipy)
- gensim >= 4.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- requests >= 2.31.0
- tqdm >= 4.65.0

### Optional Packages
- lime >= 0.2.0 (XAI)
- shap >= 0.42.0 (XAI)
- sentence-transformers >= 2.2.0 (Advanced embeddings)
- transformers >= 4.30.0 (Hugging Face)
- torch >= 2.0.0 (PyTorch)
- xgboost >= 1.7.0 (Gradient boosting)

## 🎯 What Works Without Optional Packages

### Core Functionality (Always Available)
✅ TF-IDF vectorization  
✅ Doc2Vec embeddings  
✅ Basic classifiers (SVM, MLP, Random Forest)  
✅ Data visualization  
✅ Pipeline execution  
✅ Gemini summarization  

### Limited Functionality (Without Optional Packages)
⚠️ XAI explanations (LIME/SHAP) - will be skipped  
⚠️ Sentence-BERT vectors - will use Doc2Vec only  
⚠️ Advanced classifiers (XGBoost) - will use basic ones  

## 🚀 Quick Start After Installation

```bash
# 1. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

# 2. Run quick test
python run_pipeline.py --quick

# 3. Run full pipeline
python run_pipeline.py
```

## 📞 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Ensure you're using scikit-learn 1.2.2
3. Verify Ollama is running (for Gemini features)
4. Try the quick mode first: `python run_pipeline.py --quick` 