"""
20 Newsgroups Dataset Class for Fully Supervised Learning
This class provides methods to prepare and manage the 20 Newsgroups dataset
for traditional supervised learning experiments.
"""

from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random

class NewsgroupsDataset:
    """
    Class for managing 20 Newsgroups dataset for supervised learning
    """
    
    def __init__(self, remove_headers=True, remove_footers=True, remove_quotes=True, 
                 random_state=42, test_size=0.2, val_size=0.1):
        """
        Initialize the NewsgroupsDataset
        
        Args:
            remove_headers (bool): Remove headers from posts
            remove_footers (bool): Remove footers from posts
            remove_quotes (bool): Remove quoted text from posts
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of training data for validation set
        """
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.remove_quotes = remove_quotes
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        
        # Initialize data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.class_names = None
        self.num_classes = None
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the 20 Newsgroups dataset"""
        print("Loading 20 Newsgroups dataset...")
        
        # Define what to remove
        remove = []
        if self.remove_headers:
            remove.append('headers')
        if self.remove_footers:
            remove.append('footers')
        if self.remove_quotes:
            remove.append('quotes')
        
        # Load training and test data
        newsgroups_train = fetch_20newsgroups(
            subset='train', 
            remove=tuple(remove) if remove else None,
            random_state=self.random_state
        )
        
        newsgroups_test = fetch_20newsgroups(
            subset='test', 
            remove=tuple(remove) if remove else None,
            random_state=self.random_state
        )
        
        # Store class names
        self.class_names = newsgroups_train.target_names
        self.num_classes = len(self.class_names)
        
        # Combine train and test for custom split
        X_combined = newsgroups_train.data + newsgroups_test.data
        y_combined = np.concatenate([newsgroups_train.target, newsgroups_test.target])
        
        # Create train/test split
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_combined, y_combined, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_combined
        )
        
        # Create train/validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"Dataset loaded successfully!")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Validation samples: {len(self.X_val)}")
        print(f"  - Test samples: {len(self.X_test)}")
        print(f"  - Total classes: {self.num_classes}")
    
    def get_train_data(self):
        """Get training data"""
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        """Get validation data"""
        return self.X_val, self.y_val
    
    def get_test_data(self):
        """Get test data"""
        return self.X_test, self.y_test
    
    def get_all_data(self):
        """Get all data splits"""
        return {
            'train': (self.X_train, self.y_train),
            'validation': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test)
        }
    
    def get_class_names(self):
        """Get class names"""
        return self.class_names
    
    def get_class_distribution(self, split='train'):
        """
        Get class distribution for a specific split
        
        Args:
            split (str): 'train', 'validation', 'test', or 'all'
        
        Returns:
            dict: Class distribution
        """
        if split == 'train':
            y_data = self.y_train
        elif split == 'validation':
            y_data = self.y_val
        elif split == 'test':
            y_data = self.y_test
        elif split == 'all':
            y_data = np.concatenate([self.y_train, self.y_val, self.y_test])
        else:
            raise ValueError("split must be 'train', 'validation', 'test', or 'all'")
        
        class_counts = Counter(y_data)
        return {self.class_names[i]: count for i, count in class_counts.items()}
    
    def get_balanced_subset(self, samples_per_class=100, split='train', random_state=None):
        """
        Get a balanced subset of the dataset
        
        Args:
            samples_per_class (int): Number of samples per class
            split (str): Which split to sample from
            random_state (int): Random seed for sampling
        
        Returns:
            tuple: (X_subset, y_subset)
        """
        if split == 'train':
            X_data, y_data = self.X_train, self.y_train
        elif split == 'validation':
            X_data, y_data = self.X_val, self.y_val
        elif split == 'test':
            X_data, y_data = self.X_test, self.y_test
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        
        X_subset = []
        y_subset = []
        
        for class_idx in range(self.num_classes):
            # Get indices for current class
            class_indices = [i for i, label in enumerate(y_data) if label == class_idx]
            
            if len(class_indices) >= samples_per_class:
                # Randomly sample
                selected_indices = random.sample(class_indices, samples_per_class)
            else:
                # Use all available samples
                selected_indices = class_indices
                print(f"Warning: Class {self.class_names[class_idx]} has only {len(class_indices)} samples")
            
            X_subset.extend([X_data[i] for i in selected_indices])
            y_subset.extend([y_data[i] for i in selected_indices])
        
        return X_subset, np.array(y_subset)
    
    def get_text_statistics(self, split='train'):
        """
        Get text statistics for a specific split
        
        Args:
            split (str): 'train', 'validation', 'test', or 'all'
        
        Returns:
            dict: Text statistics
        """
        if split == 'train':
            X_data = self.X_train
        elif split == 'validation':
            X_data = self.X_val
        elif split == 'test':
            X_data = self.X_test
        elif split == 'all':
            X_data = self.X_train + self.X_val + self.X_test
        else:
            raise ValueError("split must be 'train', 'validation', 'test', or 'all'")
        
        # Calculate text lengths
        text_lengths = [len(text.split()) for text in X_data]
        char_lengths = [len(text) for text in X_data]
        
        stats = {
            'num_samples': len(X_data),
            'avg_words': np.mean(text_lengths),
            'median_words': np.median(text_lengths),
            'min_words': np.min(text_lengths),
            'max_words': np.max(text_lengths),
            'std_words': np.std(text_lengths),
            'avg_chars': np.mean(char_lengths),
            'median_chars': np.median(char_lengths),
            'min_chars': np.min(char_lengths),
            'max_chars': np.max(char_lengths),
            'std_chars': np.std(char_lengths)
        }
        
        return stats
    
    def create_dataframe(self, split='train'):
        """
        Create a pandas DataFrame for a specific split
        
        Args:
            split (str): 'train', 'validation', 'test', or 'all'
        
        Returns:
            pd.DataFrame: DataFrame with text and label columns
        """
        if split == 'train':
            X_data, y_data = self.X_train, self.y_train
        elif split == 'validation':
            X_data, y_data = self.X_val, self.y_val
        elif split == 'test':
            X_data, y_data = self.X_test, self.y_test
        elif split == 'all':
            X_data = self.X_train + self.X_val + self.X_test
            y_data = np.concatenate([self.y_train, self.y_val, self.y_test])
        else:
            raise ValueError("split must be 'train', 'validation', 'test', or 'all'")
        
        df = pd.DataFrame({
            'text': X_data,
            'label': y_data,
            'class_name': [self.class_names[label] for label in y_data]
        })
        
        return df
    
    def get_sample_texts(self, num_samples=5, split='train'):
        """
        Get sample texts from the dataset
        
        Args:
            num_samples (int): Number of samples per class
            split (str): Which split to sample from
        
        Returns:
            dict: Sample texts organized by class
        """
        if split == 'train':
            X_data, y_data = self.X_train, self.y_train
        elif split == 'validation':
            X_data, y_data = self.X_val, self.y_val
        elif split == 'test':
            X_data, y_data = self.X_test, self.y_test
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        
        samples = {}
        
        for class_idx in range(self.num_classes):
            class_indices = [i for i, label in enumerate(y_data) if label == class_idx]
            
            if len(class_indices) >= num_samples:
                selected_indices = random.sample(class_indices, num_samples)
            else:
                selected_indices = class_indices
            
            samples[self.class_names[class_idx]] = [
                X_data[i] for i in selected_indices
            ]
        
        return samples
    
    def print_summary(self):
        """Print a summary of the dataset"""
        print("\n" + "="*60)
        print("20 NEWSCROUPS DATASET SUMMARY")
        print("="*60)
        
        print(f"Total classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        
        print(f"\nData splits:")
        print(f"  Training: {len(self.X_train)} samples")
        print(f"  Validation: {len(self.X_val)} samples")
        print(f"  Test: {len(self.X_test)} samples")
        print(f"  Total: {len(self.X_train) + len(self.X_val) + len(self.X_test)} samples")
        
        # Class distribution
        train_dist = self.get_class_distribution('train')
        print(f"\nClass distribution (training set):")
        for class_name, count in train_dist.items():
            percentage = (count / len(self.X_train)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Text statistics
        train_stats = self.get_text_statistics('train')
        print(f"\nText statistics (training set):")
        print(f"  Average words per text: {train_stats['avg_words']:.1f}")
        print(f"  Average characters per text: {train_stats['avg_chars']:.1f}")
        print(f"  Min words: {train_stats['min_words']}")
        print(f"  Max words: {train_stats['max_words']}")
        
        print("\n" + "="*60)

def create_newsgroups_dataset(**kwargs):
    """
    Factory function to create a NewsgroupsDataset instance
    
    Args:
        **kwargs: Arguments to pass to NewsgroupsDataset constructor
    
    Returns:
        NewsgroupsDataset: Configured dataset instance
    """
    return NewsgroupsDataset(**kwargs)

# Example usage
if __name__ == "__main__":
    # Create dataset instance
    dataset = NewsgroupsDataset(
        remove_headers=True,
        remove_footers=True,
        remove_quotes=True,
        random_state=42,
        test_size=0.2,
        val_size=0.1
    )
    
    # Print summary
    dataset.print_summary()
    
    # Get data splits
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_validation_data()
    X_test, y_test = dataset.get_test_data()
    
    print(f"\nData shapes:")
    print(f"X_train: {len(X_train)} samples")
    print(f"X_val: {len(X_val)} samples")
    print(f"X_test: {len(X_test)} samples")
    
    # Get balanced subset
    X_balanced, y_balanced = dataset.get_balanced_subset(samples_per_class=50, split='train')
    print(f"\nBalanced subset: {len(X_balanced)} samples")
    
    # Create DataFrame
    df_train = dataset.create_dataframe('train')
    print(f"\nTraining DataFrame shape: {df_train.shape}")
    print(f"Columns: {df_train.columns.tolist()}")
    
    # Get sample texts
    samples = dataset.get_sample_texts(num_samples=2, split='train')
    print(f"\nSample texts (first 2 classes):")
    for i, (class_name, texts) in enumerate(samples.items()):
        if i >= 2:  # Show only first 2 classes
            break
        print(f"\n{class_name}:")
        for j, text in enumerate(texts):
            print(f"  Sample {j+1}: {text[:100]}...") 