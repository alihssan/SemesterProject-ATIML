"""
AG News Dataset Class for Fully Supervised Learning and Few-Shot Learning
This class provides methods to prepare and manage the AG News dataset
for both traditional supervised learning and few-shot learning experiments.
"""

from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random

class AGNewsDataset:
    """
    Class for managing AG News dataset for supervised learning and few-shot learning
    """
    
    def __init__(self, random_state=42, test_size=0.2, val_size=0.1, 
                 few_shot_shots=[1, 3, 5, 10]):
        """
        Initialize the AGNewsDataset
        
        Args:
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of training data for validation set
            few_shot_shots (list): List of shot counts for few-shot learning
        """
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.few_shot_shots = few_shot_shots
        
        # Initialize data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.class_names = None
        self.num_classes = None
        
        # Few-shot data containers
        self.few_shot_data = {}
        
        # Load dataset
        self._load_dataset()
        self._prepare_few_shot_data()
    
    def _load_dataset(self):
        """Load the AG News dataset"""
        print("Loading AG News dataset...")
        
        # Load dataset from Hugging Face
        ag_news_dataset = load_dataset("ag_news")
        
        # Convert to pandas DataFrames for easier handling
        ag_news_train = pd.DataFrame(ag_news_dataset['train'])
        ag_news_test = pd.DataFrame(ag_news_dataset['test'])
        
        # Store class names
        self.class_names = ag_news_dataset['train'].features['label'].names
        self.num_classes = len(self.class_names)
        
        # Extract data and labels
        X_train_full = ag_news_train['text'].tolist()
        y_train_full = ag_news_train['label'].tolist()
        X_test_full = ag_news_test['text'].tolist()
        y_test_full = ag_news_test['label'].tolist()
        
        # Combine train and test for custom split
        X_combined = X_train_full + X_test_full
        y_combined = np.array(y_train_full + y_test_full)
        
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
        print(f"  - Class names: {self.class_names}")
    
    def _prepare_few_shot_data(self):
        """Prepare few-shot learning datasets"""
        print("\nPreparing few-shot learning datasets...")
        
        for shots in self.few_shot_shots:
            print(f"Creating {shots}-shot setup...")
            
            # Sample few examples per class for training
            X_few_train = []
            y_few_train = []
            
            for class_idx in range(self.num_classes):
                # Get indices for current class
                class_indices = [i for i, label in enumerate(self.y_train) if label == class_idx]
                
                if len(class_indices) >= shots:
                    # Randomly sample 'shots' examples
                    selected_indices = random.sample(class_indices, shots)
                    X_few_train.extend([self.X_train[i] for i in selected_indices])
                    y_few_train.extend([self.y_train[i] for i in selected_indices])
                else:
                    # If not enough examples, use all available
                    X_few_train.extend([self.X_train[i] for i in class_indices])
                    y_few_train.extend([self.y_train[i] for i in class_indices])
                    print(f"Warning: Class {self.class_names[class_idx]} has only {len(class_indices)} examples")
            
            self.few_shot_data[f'{shots}_shot'] = {
                'X_train': X_few_train,
                'y_train': np.array(y_few_train),
                'X_test': self.X_test,
                'y_test': self.y_test,
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'shots_per_class': shots,
                'total_train_samples': len(X_few_train)
            }
            
            print(f"  {shots}-shot: {len(X_few_train)} training samples, {len(self.X_test)} test samples")
    
    def get_train_data(self):
        """Get training data for fully supervised learning"""
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        """Get validation data for fully supervised learning"""
        return self.X_val, self.y_val
    
    def get_test_data(self):
        """Get test data for fully supervised learning"""
        return self.X_test, self.y_test
    
    def get_few_shot_data(self, shots):
        """
        Get few-shot learning data for specified number of shots
        
        Args:
            shots (int): Number of shots per class (must be in self.few_shot_shots)
        
        Returns:
            dict: Few-shot dataset with train/test splits
        """
        key = f'{shots}_shot'
        if key not in self.few_shot_data:
            raise ValueError(f"Few-shot data for {shots} shots not available. Available: {self.few_shot_shots}")
        
        return self.few_shot_data[key]
    
    def get_all_few_shot_data(self):
        """Get all few-shot learning datasets"""
        return self.few_shot_data
    
    def get_all_data(self):
        """Get all data splits for fully supervised learning"""
        return {
            'train': (self.X_train, self.y_train),
            'validation': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test)
        }
    
    def get_class_names(self):
        """Get class names"""
        return self.class_names
    
    def get_class_distribution(self, split='train', few_shot_shots=None):
        """
        Get class distribution for a specific split
        
        Args:
            split (str): 'train', 'validation', 'test', 'all', or 'few_shot'
            few_shot_shots (int): Number of shots if split is 'few_shot'
        
        Returns:
            dict: Class distribution
        """
        if split == 'few_shot':
            if few_shot_shots is None:
                raise ValueError("few_shot_shots must be specified when split is 'few_shot'")
            few_shot_data = self.get_few_shot_data(few_shot_shots)
            y_data = few_shot_data['y_train']
        elif split == 'train':
            y_data = self.y_train
        elif split == 'validation':
            y_data = self.y_val
        elif split == 'test':
            y_data = self.y_test
        elif split == 'all':
            y_data = np.concatenate([self.y_train, self.y_val, self.y_test])
        else:
            raise ValueError("split must be 'train', 'validation', 'test', 'all', or 'few_shot'")
        
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
    
    def get_text_statistics(self, split='train', few_shot_shots=None):
        """
        Get text statistics for a specific split
        
        Args:
            split (str): 'train', 'validation', 'test', 'all', or 'few_shot'
            few_shot_shots (int): Number of shots if split is 'few_shot'
        
        Returns:
            dict: Text statistics
        """
        if split == 'few_shot':
            if few_shot_shots is None:
                raise ValueError("few_shot_shots must be specified when split is 'few_shot'")
            few_shot_data = self.get_few_shot_data(few_shot_shots)
            X_data = few_shot_data['X_train']
        elif split == 'train':
            X_data = self.X_train
        elif split == 'validation':
            X_data = self.X_val
        elif split == 'test':
            X_data = self.X_test
        elif split == 'all':
            X_data = self.X_train + self.X_val + self.X_test
        else:
            raise ValueError("split must be 'train', 'validation', 'test', 'all', or 'few_shot'")
        
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
    
    def create_dataframe(self, split='train', few_shot_shots=None):
        """
        Create a pandas DataFrame for a specific split
        
        Args:
            split (str): 'train', 'validation', 'test', 'all', or 'few_shot'
            few_shot_shots (int): Number of shots if split is 'few_shot'
        
        Returns:
            pd.DataFrame: DataFrame with text and label columns
        """
        if split == 'few_shot':
            if few_shot_shots is None:
                raise ValueError("few_shot_shots must be specified when split is 'few_shot'")
            few_shot_data = self.get_few_shot_data(few_shot_shots)
            X_data, y_data = few_shot_data['X_train'], few_shot_data['y_train']
        elif split == 'train':
            X_data, y_data = self.X_train, self.y_train
        elif split == 'validation':
            X_data, y_data = self.X_val, self.y_val
        elif split == 'test':
            X_data, y_data = self.X_test, self.y_test
        elif split == 'all':
            X_data = self.X_train + self.X_val + self.X_test
            y_data = np.concatenate([self.y_train, self.y_val, self.y_test])
        else:
            raise ValueError("split must be 'train', 'validation', 'test', 'all', or 'few_shot'")
        
        df = pd.DataFrame({
            'text': X_data,
            'label': y_data,
            'class_name': [self.class_names[label] for label in y_data]
        })
        
        return df
    
    def get_sample_texts(self, num_samples=5, split='train', few_shot_shots=None):
        """
        Get sample texts from the dataset
        
        Args:
            num_samples (int): Number of samples per class
            split (str): Which split to sample from
            few_shot_shots (int): Number of shots if split is 'few_shot'
        
        Returns:
            dict: Sample texts organized by class
        """
        if split == 'few_shot':
            if few_shot_shots is None:
                raise ValueError("few_shot_shots must be specified when split is 'few_shot'")
            few_shot_data = self.get_few_shot_data(few_shot_shots)
            X_data, y_data = few_shot_data['X_train'], few_shot_data['y_train']
        elif split == 'train':
            X_data, y_data = self.X_train, self.y_train
        elif split == 'validation':
            X_data, y_data = self.X_val, self.y_val
        elif split == 'test':
            X_data, y_data = self.X_test, self.y_test
        else:
            raise ValueError("split must be 'train', 'validation', 'test', or 'few_shot'")
        
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
    
    def create_prompt_format_data(self, split='train', format_type="simple", few_shot_shots=None):
        """
        Create prompt-formatted data for zero/few-shot learning
        
        Args:
            split (str): Which split to format
            format_type (str): 'simple', 'detailed', or 'question'
            few_shot_shots (int): Number of shots if split is 'few_shot'
        
        Returns:
            list: Formatted data
        """
        if split == 'few_shot':
            if few_shot_shots is None:
                raise ValueError("few_shot_shots must be specified when split is 'few_shot'")
            few_shot_data = self.get_few_shot_data(few_shot_shots)
            X_data, y_data = few_shot_data['X_train'], few_shot_data['y_train']
        elif split == 'train':
            X_data, y_data = self.X_train, self.y_train
        elif split == 'validation':
            X_data, y_data = self.X_val, self.y_val
        elif split == 'test':
            X_data, y_data = self.X_test, self.y_test
        else:
            raise ValueError("split must be 'train', 'validation', 'test', or 'few_shot'")
        
        if format_type == "simple":
            # Simple format: "Text: {text}\nLabel: {label}"
            formatted_data = []
            for text, label in zip(X_data, y_data):
                formatted_text = f"Text: {text}\nLabel: {self.class_names[label]}"
                formatted_data.append(formatted_text)
            return formatted_data
        
        elif format_type == "detailed":
            # Detailed format with class descriptions
            formatted_data = []
            for text, label in zip(X_data, y_data):
                formatted_text = f"Text: {text}\nClass: {self.class_names[label]}\nTask: Classify the above text into one of the following categories: {', '.join(self.class_names)}"
                formatted_data.append(formatted_text)
            return formatted_data
        
        elif format_type == "question":
            # Question format
            formatted_data = []
            for text, label in zip(X_data, y_data):
                formatted_text = f"Question: What category does this text belong to?\nText: {text}\nAnswer: {self.class_names[label]}"
                formatted_data.append(formatted_text)
            return formatted_data
    
    def print_summary(self):
        """Print a summary of the dataset"""
        print("\n" + "="*60)
        print("AG NEWS DATASET SUMMARY")
        print("="*60)
        
        print(f"Total classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        
        print(f"\nFully Supervised Learning Data splits:")
        print(f"  Training: {len(self.X_train)} samples")
        print(f"  Validation: {len(self.X_val)} samples")
        print(f"  Test: {len(self.X_test)} samples")
        print(f"  Total: {len(self.X_train) + len(self.X_val) + len(self.X_test)} samples")
        
        print(f"\nFew-Shot Learning Setups:")
        for shots in self.few_shot_shots:
            few_shot_data = self.few_shot_data[f'{shots}_shot']
            print(f"  {shots}-shot: {few_shot_data['total_train_samples']} training samples, {len(few_shot_data['X_test'])} test samples")
        
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

def create_ag_news_dataset(**kwargs):
    """
    Factory function to create an AGNewsDataset instance
    
    Args:
        **kwargs: Arguments to pass to AGNewsDataset constructor
    
    Returns:
        AGNewsDataset: Configured dataset instance
    """
    return AGNewsDataset(**kwargs)

# Example usage
if __name__ == "__main__":
    # Create dataset instance
    dataset = AGNewsDataset(
        random_state=42,
        test_size=0.2,
        val_size=0.1,
        few_shot_shots=[1, 3, 5, 10]
    )
    
    # Print summary
    dataset.print_summary()
    
    # Get data splits for fully supervised learning
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_validation_data()
    X_test, y_test = dataset.get_test_data()
    
    print(f"\nFully Supervised Learning Data shapes:")
    print(f"X_train: {len(X_train)} samples")
    print(f"X_val: {len(X_val)} samples")
    print(f"X_test: {len(X_test)} samples")
    
    # Get few-shot data
    few_shot_1 = dataset.get_few_shot_data(1)
    few_shot_5 = dataset.get_few_shot_data(5)
    
    print(f"\nFew-Shot Learning Data shapes:")
    print(f"1-shot: {few_shot_1['total_train_samples']} training samples")
    print(f"5-shot: {few_shot_5['total_train_samples']} training samples")
    
    # Get balanced subset
    X_balanced, y_balanced = dataset.get_balanced_subset(samples_per_class=50, split='train')
    print(f"\nBalanced subset: {len(X_balanced)} samples")
    
    # Create DataFrame
    df_train = dataset.create_dataframe('train')
    df_few_shot = dataset.create_dataframe('few_shot', few_shot_shots=3)
    print(f"\nTraining DataFrame shape: {df_train.shape}")
    print(f"3-shot DataFrame shape: {df_few_shot.shape}")
    
    # Get sample texts
    samples = dataset.get_sample_texts(num_samples=2, split='train')
    print(f"\nSample texts (first 2 classes):")
    for i, (class_name, texts) in enumerate(samples.items()):
        if i >= 2:  # Show only first 2 classes
            break
        print(f"\n{class_name}:")
        for j, text in enumerate(texts):
            print(f"  Sample {j+1}: {text[:100]}...")
    
    # Create prompt-formatted data
    prompt_data = dataset.create_prompt_format_data(split='few_shot', format_type='simple', few_shot_shots=1)
    print(f"\nPrompt-formatted examples (1-shot):")
    for i, example in enumerate(prompt_data[:3]):
        print(f"Example {i+1}:")
        print(example[:200] + "..." if len(example) > 200 else example)
        print("-" * 50) 