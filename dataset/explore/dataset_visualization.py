"""
Dataset Visualization and Statistics Module
Comprehensive analysis of 20 Newsgroups and AG News datasets using bar charts and statistics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from newsgroups_dataset import NewsgroupsDataset
from ag_news_dataset import AGNewsDataset

class DatasetVisualizer:
    """
    Class for creating comprehensive visualizations and statistics for both datasets
    """
    
    def __init__(self, save_dir="results/visualizations"):
        """
        Initialize the visualizer
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Load datasets
        print("Loading datasets...")
        self.newsgroups = NewsgroupsDataset(random_state=42)
        self.ag_news = AGNewsDataset(random_state=42)
        
        print("Datasets loaded successfully!")
    
    def create_class_distribution_comparison(self):
        """Create bar chart comparing class distributions"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 20 Newsgroups class distribution
        ng_dist = self.newsgroups.get_class_distribution('train')
        ng_classes = list(ng_dist.keys())
        ng_counts = list(ng_dist.values())
        
        bars1 = ax1.bar(range(len(ng_classes)), ng_counts, color='skyblue', alpha=0.7)
        ax1.set_title('20 Newsgroups - Class Distribution (Training Set)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(len(ng_classes)))
        ax1.set_xticklabels(ng_classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars1, ng_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(ng_counts),
                    str(count), ha='center', va='bottom', fontsize=8)
        
        # AG News class distribution
        ag_dist = self.ag_news.get_class_distribution('train')
        ag_classes = list(ag_dist.keys())
        ag_counts = list(ag_dist.values())
        
        bars2 = ax2.bar(range(len(ag_classes)), ag_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('AG News - Class Distribution (Training Set)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks(range(len(ag_classes)))
        ax2.set_xticklabels(ag_classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars2, ag_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(ag_counts),
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/class_distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution comparison saved: {self.save_dir}/class_distribution_comparison.png")
    
    def create_text_length_comparison(self):
        """Create bar charts comparing text length statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get text statistics
        ng_train_stats = self.newsgroups.get_text_statistics('train')
        ng_test_stats = self.newsgroups.get_text_statistics('test')
        ag_train_stats = self.ag_news.get_text_statistics('train')
        ag_test_stats = self.ag_news.get_text_statistics('test')
        
        # 20 Newsgroups word length comparison
        ng_word_stats = [ng_train_stats['avg_words'], ng_test_stats['avg_words']]
        bars1 = ax1.bar(['Train', 'Test'], ng_word_stats, color=['skyblue', 'lightblue'])
        ax1.set_title('20 Newsgroups - Average Words per Text', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Words')
        
        # Add value labels
        for bar, value in zip(bars1, ng_word_stats):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # AG News word length comparison
        ag_word_stats = [ag_train_stats['avg_words'], ag_test_stats['avg_words']]
        bars2 = ax2.bar(['Train', 'Test'], ag_word_stats, color=['lightcoral', 'salmon'])
        ax2.set_title('AG News - Average Words per Text', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Words')
        
        # Add value labels
        for bar, value in zip(bars2, ag_word_stats):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 20 Newsgroups character length comparison
        ng_char_stats = [ng_train_stats['avg_chars'], ng_test_stats['avg_chars']]
        bars3 = ax3.bar(['Train', 'Test'], ng_char_stats, color=['skyblue', 'lightblue'])
        ax3.set_title('20 Newsgroups - Average Characters per Text', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Characters')
        
        # Add value labels
        for bar, value in zip(bars3, ng_char_stats):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # AG News character length comparison
        ag_char_stats = [ag_train_stats['avg_chars'], ag_test_stats['avg_chars']]
        bars4 = ax4.bar(['Train', 'Test'], ag_char_stats, color=['lightcoral', 'salmon'])
        ax4.set_title('AG News - Average Characters per Text', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Characters')
        
        # Add value labels
        for bar, value in zip(bars4, ag_char_stats):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/text_length_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Text length comparison saved: {self.save_dir}/text_length_comparison.png")
    
    def create_dataset_size_comparison(self):
        """Create bar chart comparing dataset sizes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Dataset sizes
        ng_sizes = [len(self.newsgroups.X_train), len(self.newsgroups.X_val), len(self.newsgroups.X_test)]
        ag_sizes = [len(self.ag_news.X_train), len(self.ag_news.X_val), len(self.ag_news.X_test)]
        
        # 20 Newsgroups sizes
        bars1 = ax1.bar(['Train', 'Validation', 'Test'], ng_sizes, color=['skyblue', 'lightblue', 'powderblue'])
        ax1.set_title('20 Newsgroups - Dataset Sizes', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        
        # Add value labels
        for bar, value in zip(bars1, ng_sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{value:,}', ha='center', va='bottom')
        
        # AG News sizes
        bars2 = ax2.bar(['Train', 'Validation', 'Test'], ag_sizes, color=['lightcoral', 'salmon', 'mistyrose'])
        ax2.set_title('AG News - Dataset Sizes', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Samples')
        
        # Add value labels
        for bar, value in zip(bars2, ag_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{value:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/dataset_size_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dataset size comparison saved: {self.save_dir}/dataset_size_comparison.png")
    
    def create_few_shot_analysis(self):
        """Create bar chart showing few-shot learning setups for AG News"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get few-shot data
        few_shot_data = self.ag_news.get_all_few_shot_data()
        
        shots = []
        train_samples = []
        test_samples = []
        
        for key, data in few_shot_data.items():
            shot_num = int(key.split('_')[0])
            shots.append(shot_num)
            train_samples.append(data['total_train_samples'])
            test_samples.append(len(data['X_test']))
        
        # Sort by shots
        sorted_indices = np.argsort(shots)
        shots = [shots[i] for i in sorted_indices]
        train_samples = [train_samples[i] for i in sorted_indices]
        test_samples = [test_samples[i] for i in sorted_indices]
        
        x = np.arange(len(shots))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_samples, width, label='Training Samples', color='lightcoral', alpha=0.7)
        bars2 = ax.bar(x + width/2, test_samples, width, label='Test Samples', color='salmon', alpha=0.7)
        
        ax.set_title('AG News - Few-Shot Learning Setups', fontsize=14, fontweight='bold')
        ax.set_xlabel('Shots per Class')
        ax.set_ylabel('Number of Samples')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{shot}-shot' for shot in shots])
        ax.legend()
        
        # Add value labels
        for bar, value in zip(bars1, train_samples):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(value), ha='center', va='bottom', fontsize=10)
        
        for bar, value in zip(bars2, test_samples):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(value), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/few_shot_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Few-shot analysis saved: {self.save_dir}/few_shot_analysis.png")
    
    def create_statistics_summary_table(self):
        """Create a comprehensive statistics summary"""
        # Collect all statistics
        ng_train_stats = self.newsgroups.get_text_statistics('train')
        ng_test_stats = self.newsgroups.get_text_statistics('test')
        ag_train_stats = self.ag_news.get_text_statistics('train')
        ag_test_stats = self.ag_news.get_text_statistics('test')
        
        # Create summary DataFrame
        summary_data = {
            'Metric': [
                'Total Classes',
                'Training Samples',
                'Validation Samples', 
                'Test Samples',
                'Total Samples',
                'Avg Words (Train)',
                'Avg Words (Test)',
                'Avg Chars (Train)',
                'Avg Chars (Test)',
                'Min Words (Train)',
                'Max Words (Train)',
                'Std Words (Train)'
            ],
            '20 Newsgroups': [
                self.newsgroups.num_classes,
                len(self.newsgroups.X_train),
                len(self.newsgroups.X_val),
                len(self.newsgroups.X_test),
                len(self.newsgroups.X_train) + len(self.newsgroups.X_val) + len(self.newsgroups.X_test),
                f"{ng_train_stats['avg_words']:.1f}",
                f"{ng_test_stats['avg_words']:.1f}",
                f"{ng_train_stats['avg_chars']:.1f}",
                f"{ng_test_stats['avg_chars']:.1f}",
                ng_train_stats['min_words'],
                ng_train_stats['max_words'],
                f"{ng_train_stats['std_words']:.1f}"
            ],
            'AG News': [
                self.ag_news.num_classes,
                len(self.ag_news.X_train),
                len(self.ag_news.X_val),
                len(self.ag_news.X_test),
                len(self.ag_news.X_train) + len(self.ag_news.X_val) + len(self.ag_news.X_test),
                f"{ag_train_stats['avg_words']:.1f}",
                f"{ag_test_stats['avg_words']:.1f}",
                f"{ag_train_stats['avg_chars']:.1f}",
                f"{ag_test_stats['avg_chars']:.1f}",
                ag_train_stats['min_words'],
                ag_train_stats['max_words'],
                f"{ag_train_stats['std_words']:.1f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv(f"{self.save_dir}/dataset_statistics_summary.csv", index=False)
        
        # Create a nice formatted table
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                        cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(summary_df) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Dataset Statistics Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{self.save_dir}/statistics_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistics summary saved: {self.save_dir}/dataset_statistics_summary.csv")
        print(f"Statistics table saved: {self.save_dir}/statistics_summary_table.png")
        
        return summary_df
    
    def create_class_balance_analysis(self):
        """Create analysis of class balance for both datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 20 Newsgroups class balance
        ng_dist = self.newsgroups.get_class_distribution('train')
        ng_classes = list(ng_dist.keys())
        ng_counts = list(ng_dist.values())
        ng_percentages = [(count / sum(ng_counts)) * 100 for count in ng_counts]
        
        bars1 = ax1.bar(range(len(ng_classes)), ng_percentages, color='skyblue', alpha=0.7)
        ax1.set_title('20 Newsgroups - Class Balance (%)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Percentage of Total Samples')
        ax1.set_xticks(range(len(ng_classes)))
        ax1.set_xticklabels(ng_classes, rotation=45, ha='right')
        
        # Add percentage labels
        for bar, percentage in zip(bars1, ng_percentages):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # AG News class balance
        ag_dist = self.ag_news.get_class_distribution('train')
        ag_classes = list(ag_dist.keys())
        ag_counts = list(ag_dist.values())
        ag_percentages = [(count / sum(ag_counts)) * 100 for count in ag_counts]
        
        bars2 = ax2.bar(range(len(ag_classes)), ag_percentages, color='lightcoral', alpha=0.7)
        ax2.set_title('AG News - Class Balance (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Percentage of Total Samples')
        ax2.set_xticks(range(len(ag_classes)))
        ax2.set_xticklabels(ag_classes, rotation=45, ha='right')
        
        # Add percentage labels
        for bar, percentage in zip(bars2, ag_percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/class_balance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class balance analysis saved: {self.save_dir}/class_balance_analysis.png")
    
    def run_complete_analysis(self):
        """Run all visualizations and analyses"""
        print("🚀 Starting comprehensive dataset analysis...")
        print("="*60)
        
        # Create all visualizations
        print("1. Creating class distribution comparison...")
        self.create_class_distribution_comparison()
        
        print("2. Creating text length comparison...")
        self.create_text_length_comparison()
        
        print("3. Creating dataset size comparison...")
        self.create_dataset_size_comparison()
        
        print("4. Creating few-shot analysis...")
        self.create_few_shot_analysis()
        
        print("5. Creating class balance analysis...")
        self.create_class_balance_analysis()
        
        print("6. Creating statistics summary...")
        summary_df = self.create_statistics_summary_table()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All visualizations saved in: {self.save_dir}/")
        print("\nGenerated files:")
        print("  📊 class_distribution_comparison.png")
        print("  📊 text_length_comparison.png")
        print("  📊 dataset_size_comparison.png")
        print("  📊 few_shot_analysis.png")
        print("  📊 class_balance_analysis.png")
        print("  📊 statistics_summary_table.png")
        print("  📄 dataset_statistics_summary.csv")
        
        # Print key insights
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        
        ng_total = len(self.newsgroups.X_train) + len(self.newsgroups.X_val) + len(self.newsgroups.X_test)
        ag_total = len(self.ag_news.X_train) + len(self.ag_news.X_val) + len(self.ag_news.X_test)
        
        print(f"📈 Dataset Sizes:")
        print(f"  20 Newsgroups: {ng_total:,} total samples ({self.newsgroups.num_classes} classes)")
        print(f"  AG News: {ag_total:,} total samples ({self.ag_news.num_classes} classes)")
        
        ng_train_stats = self.newsgroups.get_text_statistics('train')
        ag_train_stats = self.ag_news.get_text_statistics('train')
        
        print(f"\n📝 Text Characteristics:")
        print(f"  20 Newsgroups: {ng_train_stats['avg_words']:.1f} avg words, {ng_train_stats['avg_chars']:.1f} avg chars")
        print(f"  AG News: {ag_train_stats['avg_words']:.1f} avg words, {ag_train_stats['avg_chars']:.1f} avg chars")
        
        print(f"\n🎯 Few-Shot Learning:")
        few_shot_data = self.ag_news.get_all_few_shot_data()
        for key, data in few_shot_data.items():
            print(f"  {key}: {data['total_train_samples']} training samples")
        
        return summary_df

def main():
    """Main function to run the complete analysis"""
    visualizer = DatasetVisualizer()
    summary_df = visualizer.run_complete_analysis()
    
    # Display summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main() 