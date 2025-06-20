"""
Data Cleaning Module
Provides functions to clean and normalize 20 Newsgroups and AG News datasets
"""

import re
import string
import unicodedata
from typing import List, Dict, Any
import numpy as np

class DataCleaner:
    """
    Class for cleaning and normalizing text data
    """
    
    def __init__(self):
        """Initialize the data cleaner"""
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.unicode_pattern = re.compile(r'[^\x00-\x7F]+')
        
    def clean_text(self, text: str, 
                   remove_html: bool = True,
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   normalize_whitespace: bool = True,
                   to_lowercase: bool = True,
                   normalize_unicode: bool = True,
                   normalize_punctuation: bool = True) -> str:
        """
        Clean a single text sample
        
        Args:
            text (str): Input text
            remove_html (bool): Remove HTML tags
            remove_urls (bool): Remove URLs
            remove_emails (bool): Remove email addresses
            normalize_whitespace (bool): Normalize whitespace
            to_lowercase (bool): Convert to lowercase
            normalize_unicode (bool): Normalize Unicode characters
            normalize_punctuation (bool): Normalize punctuation
        
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        if remove_html:
            text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub(' [URL] ', text)
        
        # Remove email addresses
        if remove_emails:
            text = self.email_pattern.sub(' [EMAIL] ', text)
        
        # Normalize Unicode characters
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase
        if to_lowercase:
            text = text.lower()
        
        # Normalize punctuation
        if normalize_punctuation:
            # Replace multiple punctuation marks with single
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{2,}', '.', text)
            # Standardize quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
        
        # Normalize whitespace
        if normalize_whitespace:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            # Remove leading/trailing whitespace
            text = text.strip()
        
        return text
    
    def filter_texts(self, texts: List[str], 
                    min_words: int = 10,
                    max_words: int = 1000,
                    remove_empty: bool = True) -> List[str]:
        """
        Filter texts based on length criteria
        
        Args:
            texts (List[str]): List of texts
            min_words (int): Minimum number of words
            max_words (int): Maximum number of words
            remove_empty (bool): Remove empty texts
        
        Returns:
            List[str]: Filtered texts
        """
        filtered_texts = []
        
        for text in texts:
            word_count = len(text.split())
            
            # Skip empty texts
            if remove_empty and word_count == 0:
                continue
            
            # Skip very short texts
            if word_count < min_words:
                continue
            
            # Skip very long texts
            if word_count > max_words:
                continue
            
            filtered_texts.append(text)
        
        return filtered_texts
    
    def clean_dataset(self, texts: List[str], 
                     labels: List[int] = None,
                     cleaning_config: Dict[str, Any] = None) -> tuple:
        """
        Clean an entire dataset
        
        Args:
            texts (List[str]): List of texts
            labels (List[int]): List of labels (optional)
            cleaning_config (Dict): Cleaning configuration
        
        Returns:
            tuple: (cleaned_texts, cleaned_labels, cleaning_stats)
        """
        if cleaning_config is None:
            cleaning_config = {
                'remove_html': True,
                'remove_urls': True,
                'remove_emails': True,
                'normalize_whitespace': True,
                'to_lowercase': True,
                'normalize_unicode': True,
                'normalize_punctuation': True,
                'min_words': 10,
                'max_words': 1000,
                'remove_empty': True
            }
        
        print(f"Cleaning {len(texts)} texts...")
        
        # Clean texts
        cleaned_texts = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(texts)} texts...")
            
            cleaned_text = self.clean_text(text, **{k: v for k, v in cleaning_config.items() 
                                                   if k in ['remove_html', 'remove_urls', 'remove_emails', 
                                                           'normalize_whitespace', 'to_lowercase', 
                                                           'normalize_unicode', 'normalize_punctuation']})
            cleaned_texts.append(cleaned_text)
        
        # Filter texts
        filtered_texts = self.filter_texts(cleaned_texts, 
                                         min_words=cleaning_config.get('min_words', 10),
                                         max_words=cleaning_config.get('max_words', 1000),
                                         remove_empty=cleaning_config.get('remove_empty', True))
        
        # Filter labels if provided
        cleaned_labels = None
        if labels is not None:
            # Keep labels for texts that passed filtering
            filtered_indices = []
            for i, text in enumerate(cleaned_texts):
                if text in filtered_texts:
                    filtered_indices.append(i)
            
            cleaned_labels = [labels[i] for i in filtered_indices]
        
        # Calculate cleaning statistics
        cleaning_stats = {
            'original_count': len(texts),
            'cleaned_count': len(filtered_texts),
            'removed_count': len(texts) - len(filtered_texts),
            'removal_percentage': ((len(texts) - len(filtered_texts)) / len(texts)) * 100
        }
        
        print(f"Cleaning complete!")
        print(f"  Original texts: {cleaning_stats['original_count']}")
        print(f"  Cleaned texts: {cleaning_stats['cleaned_count']}")
        print(f"  Removed texts: {cleaning_stats['removed_count']} ({cleaning_stats['removal_percentage']:.2f}%)")
        
        return filtered_texts, cleaned_labels, cleaning_stats
    
    def create_cleaning_pipeline(self, pipeline_name: str = "standard") -> Dict[str, Any]:
        """
        Create predefined cleaning pipelines
        
        Args:
            pipeline_name (str): Name of the pipeline
        
        Returns:
            Dict: Cleaning configuration
        """
        pipelines = {
            "minimal": {
                'remove_html': True,
                'remove_urls': False,
                'remove_emails': False,
                'normalize_whitespace': True,
                'to_lowercase': False,
                'normalize_unicode': False,
                'normalize_punctuation': False,
                'min_words': 5,
                'max_words': 2000,
                'remove_empty': True
            },
            "standard": {
                'remove_html': True,
                'remove_urls': True,
                'remove_emails': True,
                'normalize_whitespace': True,
                'to_lowercase': True,
                'normalize_unicode': True,
                'normalize_punctuation': True,
                'min_words': 10,
                'max_words': 1000,
                'remove_empty': True
            },
            "aggressive": {
                'remove_html': True,
                'remove_urls': True,
                'remove_emails': True,
                'normalize_whitespace': True,
                'to_lowercase': True,
                'normalize_unicode': True,
                'normalize_punctuation': True,
                'min_words': 20,
                'max_words': 500,
                'remove_empty': True
            }
        }
        
        return pipelines.get(pipeline_name, pipelines["standard"])
    
    def compare_cleaning_effects(self, texts: List[str], 
                               sample_size: int = 100) -> Dict[str, Any]:
        """
        Compare the effects of different cleaning approaches
        
        Args:
            texts (List[str]): List of texts
            sample_size (int): Number of texts to sample
        
        Returns:
            Dict: Comparison results
        """
        # Sample texts
        sample_texts = texts[:sample_size]
        
        # Apply different cleaning approaches
        pipelines = ["minimal", "standard", "aggressive"]
        results = {}
        
        for pipeline_name in pipelines:
            config = self.create_cleaning_pipeline(pipeline_name)
            cleaned_texts, _, stats = self.clean_dataset(sample_texts, cleaning_config=config)
            
            # Calculate additional metrics
            avg_length = np.mean([len(text.split()) for text in cleaned_texts]) if cleaned_texts else 0
            unique_words = len(set(' '.join(cleaned_texts).split())) if cleaned_texts else 0
            
            results[pipeline_name] = {
                'config': config,
                'stats': stats,
                'avg_length': avg_length,
                'unique_words': unique_words,
                'sample_texts': cleaned_texts[:3]  # First 3 cleaned texts
            }
        
        return results

def clean_newsgroups_dataset(dataset, pipeline_name: str = "standard"):
    """
    Clean 20 Newsgroups dataset
    
    Args:
        dataset: NewsgroupsDataset instance
        pipeline_name (str): Cleaning pipeline to use
    
    Returns:
        tuple: (cleaned_X_train, cleaned_y_train, cleaned_X_val, cleaned_y_val, cleaned_X_test, cleaned_y_test)
    """
    cleaner = DataCleaner()
    config = cleaner.create_cleaning_pipeline(pipeline_name)
    
    print("Cleaning 20 Newsgroups dataset...")
    
    # Clean training data
    X_train, y_train, train_stats = cleaner.clean_dataset(dataset.X_train, dataset.y_train, config)
    
    # Clean validation data
    X_val, y_val, val_stats = cleaner.clean_dataset(dataset.X_val, dataset.y_val, config)
    
    # Clean test data
    X_test, y_test, test_stats = cleaner.clean_dataset(dataset.X_test, dataset.y_test, config)
    
    print(f"\nCleaning Summary:")
    print(f"  Training: {train_stats['original_count']} → {train_stats['cleaned_count']} ({train_stats['removal_percentage']:.2f}% removed)")
    print(f"  Validation: {val_stats['original_count']} → {val_stats['cleaned_count']} ({val_stats['removal_percentage']:.2f}% removed)")
    print(f"  Test: {test_stats['original_count']} → {test_stats['cleaned_count']} ({test_stats['removal_percentage']:.2f}% removed)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def clean_ag_news_dataset(dataset, pipeline_name: str = "standard"):
    """
    Clean AG News dataset
    
    Args:
        dataset: AGNewsDataset instance
        pipeline_name (str): Cleaning pipeline to use
    
    Returns:
        tuple: (cleaned_X_train, cleaned_y_train, cleaned_X_val, cleaned_y_val, cleaned_X_test, cleaned_y_test)
    """
    cleaner = DataCleaner()
    config = cleaner.create_cleaning_pipeline(pipeline_name)
    
    print("Cleaning AG News dataset...")
    
    # Clean training data
    X_train, y_train, train_stats = cleaner.clean_dataset(dataset.X_train, dataset.y_train, config)
    
    # Clean validation data
    X_val, y_val, val_stats = cleaner.clean_dataset(dataset.X_val, dataset.y_val, config)
    
    # Clean test data
    X_test, y_test, test_stats = cleaner.clean_dataset(dataset.X_test, dataset.y_test, config)
    
    print(f"\nCleaning Summary:")
    print(f"  Training: {train_stats['original_count']} → {train_stats['cleaned_count']} ({train_stats['removal_percentage']:.2f}% removed)")
    print(f"  Validation: {val_stats['original_count']} → {val_stats['cleaned_count']} ({val_stats['removal_percentage']:.2f}% removed)")
    print(f"  Test: {test_stats['original_count']} → {test_stats['cleaned_count']} ({test_stats['removal_percentage']:.2f}% removed)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Example usage
if __name__ == "__main__":
    from newsgroups_dataset import NewsgroupsDataset
    from ag_news_dataset import AGNewsDataset
    
    # Load datasets
    print("Loading datasets...")
    newsgroups = NewsgroupsDataset(random_state=42)
    ag_news = AGNewsDataset(random_state=42)
    
    # Test cleaning
    cleaner = DataCleaner()
    
    # Compare cleaning effects
    print("\nComparing cleaning effects on 20 Newsgroups...")
    ng_comparison = cleaner.compare_cleaning_effects(newsgroups.X_train[:100])
    
    print("\nComparing cleaning effects on AG News...")
    ag_comparison = cleaner.compare_cleaning_effects(ag_news.X_train[:100])
    
    # Print comparison results
    for dataset_name, comparison in [("20 Newsgroups", ng_comparison), ("AG News", ag_comparison)]:
        print(f"\n{dataset_name} Cleaning Comparison:")
        for pipeline, results in comparison.items():
            print(f"  {pipeline}: {results['stats']['cleaned_count']}/{results['stats']['original_count']} "
                  f"({results['avg_length']:.1f} avg words, {results['unique_words']} unique words)")
    
    print("\nCleaning test complete!") 