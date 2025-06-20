"""
Data Quality Analysis Module
Analyzes 20 Newsgroups and AG News datasets to identify cleaning and normalization needs
"""

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from newsgroups_dataset import NewsgroupsDataset
from ag_news_dataset import AGNewsDataset

class DataQualityAnalyzer:
    """
    Class for analyzing data quality and identifying cleaning needs
    """
    
    def __init__(self, save_dir="results/data_quality"):
        """
        Initialize the analyzer
        
        Args:
            save_dir (str): Directory to save analysis results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Load datasets
        print("Loading datasets for quality analysis...")
        self.newsgroups = NewsgroupsDataset(random_state=42)
        self.ag_news = AGNewsDataset(random_state=42)
        
        print("Datasets loaded successfully!")
    
    def analyze_text_quality(self, texts, dataset_name):
        """
        Analyze text quality issues
        
        Args:
            texts (list): List of text samples
            dataset_name (str): Name of the dataset
        
        Returns:
            dict: Quality analysis results
        """
        print(f"\nAnalyzing text quality for {dataset_name}...")
        
        quality_metrics = {
            'total_samples': len(texts),
            'empty_texts': 0,
            'very_short_texts': 0,  # < 10 words
            'very_long_texts': 0,   # > 1000 words
            'texts_with_special_chars': 0,
            'texts_with_numbers': 0,
            'texts_with_urls': 0,
            'texts_with_emails': 0,
            'texts_with_html': 0,
            'texts_with_extra_spaces': 0,
            'texts_with_unicode': 0,
            'avg_text_length': 0,
            'text_length_std': 0
        }
        
        # Patterns for detection
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        html_pattern = re.compile(r'<[^>]+>')
        unicode_pattern = re.compile(r'[^\x00-\x7F]+')
        
        text_lengths = []
        
        for text in texts:
            # Basic length analysis
            word_count = len(text.split())
            text_lengths.append(word_count)
            
            if word_count == 0:
                quality_metrics['empty_texts'] += 1
            elif word_count < 10:
                quality_metrics['very_short_texts'] += 1
            elif word_count > 1000:
                quality_metrics['very_long_texts'] += 1
            
            # Special character analysis
            if any(char in string.punctuation for char in text):
                quality_metrics['texts_with_special_chars'] += 1
            
            # Number analysis
            if any(char.isdigit() for char in text):
                quality_metrics['texts_with_numbers'] += 1
            
            # URL analysis
            if url_pattern.search(text):
                quality_metrics['texts_with_urls'] += 1
            
            # Email analysis
            if email_pattern.search(text):
                quality_metrics['texts_with_emails'] += 1
            
            # HTML analysis
            if html_pattern.search(text):
                quality_metrics['texts_with_html'] += 1
            
            # Unicode analysis
            if unicode_pattern.search(text):
                quality_metrics['texts_with_unicode'] += 1
            
            # Extra spaces analysis
            if '  ' in text or text.startswith(' ') or text.endswith(' '):
                quality_metrics['texts_with_extra_spaces'] += 1
        
        # Calculate statistics
        quality_metrics['avg_text_length'] = np.mean(text_lengths)
        quality_metrics['text_length_std'] = np.std(text_lengths)
        
        # Convert counts to percentages
        total = quality_metrics['total_samples']
        percentage_keys = []
        for key in quality_metrics:
            if key.endswith('_texts') and key != 'total_samples':
                percentage_keys.append(key)
        
        for key in percentage_keys:
            quality_metrics[f'{key}_percentage'] = (quality_metrics[key] / total) * 100
        
        return quality_metrics
    
    def analyze_class_imbalance(self, dataset_name, y_data, class_names):
        """
        Analyze class imbalance
        
        Args:
            dataset_name (str): Name of the dataset
            y_data (array): Labels
            class_names (list): Class names
        
        Returns:
            dict: Imbalance analysis results
        """
        class_counts = Counter(y_data)
        total_samples = len(y_data)
        
        imbalance_metrics = {
            'total_samples': total_samples,
            'num_classes': len(class_names),
            'class_distribution': {},
            'imbalance_ratio': 0,
            'gini_coefficient': 0
        }
        
        # Calculate class distribution
        for class_idx in range(len(class_names)):
            count = class_counts[class_idx]
            percentage = (count / total_samples) * 100
            imbalance_metrics['class_distribution'][class_names[class_idx]] = {
                'count': count,
                'percentage': percentage
            }
        
        # Calculate imbalance ratio (max/min)
        counts = list(class_counts.values())
        imbalance_metrics['imbalance_ratio'] = max(counts) / min(counts)
        
        # Calculate Gini coefficient
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        imbalance_metrics['gini_coefficient'] = gini
        
        return imbalance_metrics
    
    def create_quality_visualizations(self, ng_quality, ag_quality):
        """Create visualizations for data quality analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Text length distribution comparison
        ng_lengths = [len(text.split()) for text in self.newsgroups.X_train[:1000]]  # Sample for speed
        ag_lengths = [len(text.split()) for text in self.ag_news.X_train[:1000]]
        
        ax1.hist(ng_lengths, bins=50, alpha=0.7, label='20 Newsgroups', color='skyblue')
        ax1.hist(ag_lengths, bins=50, alpha=0.7, label='AG News', color='lightcoral')
        ax1.set_title('Text Length Distribution Comparison', fontweight='bold')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.set_xlim(0, 500)  # Limit for better visualization
        
        # Quality issues comparison
        quality_issues = ['empty_texts', 'very_short_texts', 'very_long_texts', 
                         'texts_with_special_chars', 'texts_with_numbers', 'texts_with_urls']
        
        ng_percentages = []
        ag_percentages = []
        
        for issue in quality_issues:
            ng_key = f'{issue}_percentage'
            ag_key = f'{issue}_percentage'
            ng_percentages.append(ng_quality.get(ng_key, 0))
            ag_percentages.append(ag_quality.get(ag_key, 0))
        
        x = np.arange(len(quality_issues))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, ng_percentages, width, label='20 Newsgroups', color='skyblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, ag_percentages, width, label='AG News', color='lightcoral', alpha=0.7)
        
        ax2.set_title('Quality Issues Comparison (%)', fontweight='bold')
        ax2.set_xlabel('Quality Issues')
        ax2.set_ylabel('Percentage of Samples')
        ax2.set_xticks(x)
        ax2.set_xticklabels([issue.replace('_', ' ').title() for issue in quality_issues], rotation=45, ha='right')
        ax2.legend()
        
        # Class imbalance comparison
        ng_imbalance = self.analyze_class_imbalance('20 Newsgroups', self.newsgroups.y_train, self.newsgroups.class_names)
        ag_imbalance = self.analyze_class_imbalance('AG News', self.ag_news.y_train, self.ag_news.class_names)
        
        ng_counts = [ng_imbalance['class_distribution'][name]['count'] for name in list(ng_imbalance['class_distribution'].keys())[:10]]  # First 10
        ag_counts = [ag_imbalance['class_distribution'][name]['count'] for name in ag_imbalance['class_distribution'].keys()]
        
        ax3.bar(range(len(ng_counts)), ng_counts, color='skyblue', alpha=0.7, label='20 Newsgroups')
        ax3.set_title('20 Newsgroups - Class Distribution (First 10 Classes)', fontweight='bold')
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Number of Samples')
        
        ax4.bar(range(len(ag_counts)), ag_counts, color='lightcoral', alpha=0.7, label='AG News')
        ax4.set_title('AG News - Class Distribution', fontweight='bold')
        ax4.set_xlabel('Classes')
        ax4.set_ylabel('Number of Samples')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/data_quality_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quality analysis visualization saved: {self.save_dir}/data_quality_analysis.png")
    
    def identify_cleaning_needs(self, ng_quality, ag_quality):
        """Identify specific cleaning needs based on analysis"""
        print("\n" + "="*60)
        print("CLEANING NEEDS ANALYSIS")
        print("="*60)
        
        cleaning_recommendations = {
            '20_newsgroups': [],
            'ag_news': [],
            'common': []
        }
        
        # Analyze 20 Newsgroups
        print("\n📊 20 Newsgroups Cleaning Needs:")
        if ng_quality['empty_texts'] > 0:
            cleaning_recommendations['20_newsgroups'].append("Remove empty texts")
            print(f"  ⚠️  Empty texts: {ng_quality['empty_texts']} ({ng_quality.get('empty_texts_percentage', 0):.2f}%)")
        
        if ng_quality.get('very_short_texts_percentage', 0) > 5:
            cleaning_recommendations['20_newsgroups'].append("Filter very short texts")
            print(f"  ⚠️  Very short texts: {ng_quality['very_short_texts']} ({ng_quality.get('very_short_texts_percentage', 0):.2f}%)")
        
        if ng_quality.get('texts_with_urls_percentage', 0) > 10:
            cleaning_recommendations['20_newsgroups'].append("Remove or clean URLs")
            print(f"  ⚠️  Texts with URLs: {ng_quality['texts_with_urls']} ({ng_quality.get('texts_with_urls_percentage', 0):.2f}%)")
        
        if ng_quality.get('texts_with_html_percentage', 0) > 5:
            cleaning_recommendations['20_newsgroups'].append("Remove HTML tags")
            print(f"  ⚠️  Texts with HTML: {ng_quality['texts_with_html']} ({ng_quality.get('texts_with_html_percentage', 0):.2f}%)")
        
        if ng_quality.get('texts_with_extra_spaces_percentage', 0) > 20:
            cleaning_recommendations['20_newsgroups'].append("Normalize whitespace")
            print(f"  ⚠️  Texts with extra spaces: {ng_quality['texts_with_extra_spaces']} ({ng_quality.get('texts_with_extra_spaces_percentage', 0):.2f}%)")
        
        # Analyze AG News
        print("\n📊 AG News Cleaning Needs:")
        if ag_quality['empty_texts'] > 0:
            cleaning_recommendations['ag_news'].append("Remove empty texts")
            print(f"  ⚠️  Empty texts: {ag_quality['empty_texts']} ({ag_quality.get('empty_texts_percentage', 0):.2f}%)")
        
        if ag_quality.get('very_short_texts_percentage', 0) > 5:
            cleaning_recommendations['ag_news'].append("Filter very short texts")
            print(f"  ⚠️  Very short texts: {ag_quality['very_short_texts']} ({ag_quality.get('very_short_texts_percentage', 0):.2f}%)")
        
        if ag_quality.get('texts_with_urls_percentage', 0) > 10:
            cleaning_recommendations['ag_news'].append("Remove or clean URLs")
            print(f"  ⚠️  Texts with URLs: {ag_quality['texts_with_urls']} ({ag_quality.get('texts_with_urls_percentage', 0):.2f}%)")
        
        if ag_quality.get('texts_with_html_percentage', 0) > 5:
            cleaning_recommendations['ag_news'].append("Remove HTML tags")
            print(f"  ⚠️  Texts with HTML: {ag_quality['texts_with_html']} ({ag_quality.get('texts_with_html_percentage', 0):.2f}%)")
        
        if ag_quality.get('texts_with_extra_spaces_percentage', 0) > 20:
            cleaning_recommendations['ag_news'].append("Normalize whitespace")
            print(f"  ⚠️  Texts with extra spaces: {ag_quality['texts_with_extra_spaces']} ({ag_quality.get('texts_with_extra_spaces_percentage', 0):.2f}%)")
        
        # Common recommendations
        if ng_quality.get('texts_with_unicode_percentage', 0) > 5 or ag_quality.get('texts_with_unicode_percentage', 0) > 5:
            cleaning_recommendations['common'].append("Handle Unicode characters")
            print(f"  🌐 Unicode characters detected in both datasets")
        
        if ng_quality['texts_with_special_chars_percentage'] > 50 or ag_quality['texts_with_special_chars_percentage'] > 50:
            cleaning_recommendations['common'].append("Normalize punctuation")
            print(f"  🔤 High percentage of special characters detected")
        
        return cleaning_recommendations
    
    def create_cleaning_pipeline_recommendations(self, cleaning_recommendations):
        """Create specific cleaning pipeline recommendations"""
        print("\n" + "="*60)
        print("RECOMMENDED CLEANING PIPELINE")
        print("="*60)
        
        pipeline = {
            'preprocessing_steps': [],
            'normalization_steps': [],
            'filtering_steps': []
        }
        
        # Common preprocessing steps
        pipeline['preprocessing_steps'].extend([
            "Remove HTML tags",
            "Normalize whitespace (remove extra spaces)",
            "Convert to lowercase",
            "Remove or replace URLs",
            "Handle Unicode characters"
        ])
        
        # Normalization steps
        pipeline['normalization_steps'].extend([
            "Normalize punctuation",
            "Standardize quotes and apostrophes",
            "Remove or standardize numbers",
            "Handle special characters"
        ])
        
        # Filtering steps
        pipeline['filtering_steps'].extend([
            "Remove empty texts",
            "Filter very short texts (< 10 words)",
            "Filter very long texts (> 1000 words) if needed"
        ])
        
        print("\n🔧 Recommended Preprocessing Steps:")
        for i, step in enumerate(pipeline['preprocessing_steps'], 1):
            print(f"  {i}. {step}")
        
        print("\n📏 Recommended Normalization Steps:")
        for i, step in enumerate(pipeline['normalization_steps'], 1):
            print(f"  {i}. {step}")
        
        print("\n🎯 Recommended Filtering Steps:")
        for i, step in enumerate(pipeline['filtering_steps'], 1):
            print(f"  {i}. {step}")
        
        return pipeline
    
    def run_complete_analysis(self):
        """Run complete data quality analysis"""
        print("🚀 Starting Data Quality Analysis...")
        print("="*60)
        
        # Analyze text quality
        print("1. Analyzing text quality...")
        ng_quality = self.analyze_text_quality(self.newsgroups.X_train, "20 Newsgroups")
        ag_quality = self.analyze_text_quality(self.ag_news.X_train, "AG News")
        
        # Create visualizations
        print("2. Creating quality visualizations...")
        self.create_quality_visualizations(ng_quality, ag_quality)
        
        # Identify cleaning needs
        print("3. Identifying cleaning needs...")
        cleaning_recommendations = self.identify_cleaning_needs(ng_quality, ag_quality)
        
        # Create cleaning pipeline
        print("4. Creating cleaning pipeline recommendations...")
        pipeline = self.create_cleaning_pipeline_recommendations(cleaning_recommendations)
        
        # Save results
        results = {
            '20_newsgroups_quality': ng_quality,
            'ag_news_quality': ag_quality,
            'cleaning_recommendations': cleaning_recommendations,
            'cleaning_pipeline': pipeline
        }
        
        # Save to JSON
        import json
        with open(f"{self.save_dir}/data_quality_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("✅ DATA QUALITY ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Results saved in: {self.save_dir}/")
        print("\n📊 Key Findings:")
        print(f"  20 Newsgroups: {ng_quality['total_samples']} samples analyzed")
        print(f"  AG News: {ag_quality['total_samples']} samples analyzed")
        print(f"  Quality issues identified: {len(cleaning_recommendations['common'] + cleaning_recommendations['20_newsgroups'] + cleaning_recommendations['ag_news'])}")
        
        return results

def main():
    """Main function to run the complete analysis"""
    analyzer = DataQualityAnalyzer()
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Based on the analysis, here are the key recommendations:")
    
    if results['cleaning_recommendations']['common']:
        print("\n🔧 Common cleaning needed for both datasets:")
        for rec in results['cleaning_recommendations']['common']:
            print(f"  - {rec}")
    
    if results['cleaning_recommendations']['20_newsgroups']:
        print("\n📰 20 Newsgroups specific cleaning:")
        for rec in results['cleaning_recommendations']['20_newsgroups']:
            print(f"  - {rec}")
    
    if results['cleaning_recommendations']['ag_news']:
        print("\n📰 AG News specific cleaning:")
        for rec in results['cleaning_recommendations']['ag_news']:
            print(f"  - {rec}")

if __name__ == "__main__":
    main() 