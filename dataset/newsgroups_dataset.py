import re
import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


class NewsgroupsDataset:
    def __init__(self, remove_headers=True, remove_footers=True, remove_quotes=True,
                 random_state=42, test_size=0.2, val_size=0.1, preprocess=True,
                 remove_empty=True, clip_long_docs=True, clip_percentile=95):
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.remove_quotes = remove_quotes
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.preprocess_enabled = preprocess
        self.remove_empty = remove_empty
        self.clip_long_docs = clip_long_docs
        self.clip_percentile = clip_percentile

        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")

        self.raw_texts = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.class_names = None
        self.num_classes = None

        self._load_dataset()

    def lowercase_text(self, text): return text.lower()

    def remove_non_alpha(self, text): return re.sub(r"[^a-zA-Z]", " ", text)

    def tokenize_text(self, text): return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def preprocess_text(self, text):
        text = self.lowercase_text(text)
        text = self.remove_non_alpha(text)
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        return tokens

    def _preprocess_texts(self, texts):
        return [self.preprocess_text(text) for text in texts]

    def _remove_empty_docs(self, X, y):
        return zip(*[(xi, yi) for xi, yi in zip(X, y) if len(xi) > 0])

    def _clip_long_docs(self, X, max_length):
        return [x[:max_length] for x in X]

    def _load_dataset(self):
        print("Loading 20 Newsgroups dataset...")

        remove = []
        if self.remove_headers: remove.append("headers")
        if self.remove_footers: remove.append("footers")
        if self.remove_quotes: remove.append("quotes")

        train_data = fetch_20newsgroups(subset='train', remove=tuple(remove))
        test_data = fetch_20newsgroups(subset='test', remove=tuple(remove))

        self.class_names = train_data.target_names
        self.num_classes = len(self.class_names)

        X_combined = train_data.data + test_data.data
        y_combined = np.concatenate([train_data.target, test_data.target])
        self.raw_texts = X_combined.copy()

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_combined, y_combined,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_combined
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )

        if self.preprocess_enabled:
            self.X_train = self._preprocess_texts(self.X_train)
            self.X_val = self._preprocess_texts(self.X_val)
            self.X_test = self._preprocess_texts(self.X_test)
            print("Text preprocessing applied.")

            if self.remove_empty:
                self.X_train, self.y_train = self._remove_empty_docs(self.X_train, self.y_train)
                self.X_val, self.y_val = self._remove_empty_docs(self.X_val, self.y_val)
                self.X_test, self.y_test = self._remove_empty_docs(self.X_test, self.y_test)
                print("Empty documents removed.")

            if self.clip_long_docs:
                word_lens = [len(x) for x in self.X_train]
                max_len = int(np.percentile(word_lens, self.clip_percentile))
                self.X_train = self._clip_long_docs(self.X_train, max_len)
                self.X_val = self._clip_long_docs(self.X_val, max_len)
                self.X_test = self._clip_long_docs(self.X_test, max_len)
                print(f"Documents clipped to max {max_len} tokens (at {self.clip_percentile}th percentile).")
        else:
            print("Preprocessing skipped. Raw text loaded.")

        print(f"Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")

    def get_train_data(self): return self.X_train, self.y_train
    def get_validation_data(self): return self.X_val, self.y_val
    def get_test_data(self): return self.X_test, self.y_test
    def get_class_names(self): return self.class_names

    def get_all_data(self):
        return {
            'train': (self.X_train, self.y_train),
            'validation': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test)
        }

    def get_class_distribution(self, split='train'):
        y = {'train': self.y_train, 'validation': self.y_val,
             'test': self.y_test, 'all': np.concatenate([self.y_train, self.y_val, self.y_test])}[split]
        return {self.class_names[i]: count for i, count in Counter(y).items()}

    def create_dataframe(self, split='train', raw=False):
        split_data = {
            'train': (self.X_train, self.y_train),
            'validation': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test),
            'all': (self.X_train + self.X_val + self.X_test,
                    np.concatenate([self.y_train, self.y_val, self.y_test]))
        }

        X, y = split_data[split]

        if raw:
            idx_map = {
                'train': 0,
                'validation': len(self.X_train),
                'test': len(self.X_train) + len(self.X_val),
                'all': 0
            }
            start_idx = idx_map[split]
            end_idx = start_idx + len(y)
            raw_texts = self.raw_texts[start_idx:end_idx]

            return pd.DataFrame({
                'text': raw_texts,
                'label': y,
                'class_name': [self.class_names[i] for i in y]
            })
        else:
            return pd.DataFrame({
                'data': X,
                'label': y,
                'class_name': [self.class_names[i] for i in y]
            })

    def print_summary(self):
        print("\n" + "=" * 60)
        print("20 NEWGROUPS DATASET SUMMARY")
        print("=" * 60)
        print(f"Classes: {self.num_classes}")
        print(f"Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")

        print("\n📊 Class distribution (training set):")
        dist = self.get_class_distribution('train')
        for cls, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {cls:<25} {count:>5}")
        print("=" * 60)

    def show_samples(self, num_samples=3, show_raw=False):
        print("\n📌 Sample Data (Train Set):\n")
        for i in range(num_samples):
            print(f"Sample {i + 1}")
            print(f"Class: {self.class_names[self.y_train[i]]}")
            if show_raw:
                print("Raw Text:\n", self.raw_texts[i][:500], "...\n")
            elif self.preprocess_enabled:
                print("Tokens:\n", self.X_train[i][:20], "...\n")
            else:
                print("Cleaned Text (no preprocessing):\n", self.X_train[i][:500], "...\n")

    def show_text_statistics(self, split='train'):
        if split == 'train':
            X = self.X_train
            y = self.y_train
        elif split == 'validation':
            X = self.X_val
            y = self.y_val
        elif split == 'test':
            X = self.X_test
            y = self.y_test
        elif split == 'all':
            X = self.X_train + self.X_val + self.X_test
            y = np.concatenate([self.y_train, self.y_val, self.y_test])
        else:
            raise ValueError("Invalid split")

        if self.preprocess_enabled:
            word_lengths = [len(tokens) for tokens in X]
            char_lengths = [sum(len(token) for token in tokens) for tokens in X]
        else:
            word_lengths = [len(text.split()) for text in X]
            char_lengths = [len(text) for text in X]

        print(f"\n📈 Text Statistics ({split} split):")
        print(f"  Total samples:            {len(X)}")
        print(f"  Average words/sample:     {np.mean(word_lengths):.2f}")
        print(f"  Median words/sample:      {np.median(word_lengths):.2f}")
        print(f"  25th percentile words:    {np.percentile(word_lengths, 25):.2f}")
        print(f"  75th percentile words:    {np.percentile(word_lengths, 75):.2f}")
        print(f"  Min words/sample:         {np.min(word_lengths)}")
        print(f"  Max words/sample:         {np.max(word_lengths)}")
        print(f"  Std dev (words):          {np.std(word_lengths):.2f}")
        print(f"  Average characters/sample:{np.mean(char_lengths):.2f}")

        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        plt.hist(word_lengths, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of Word Counts ({split})")
        plt.xlabel("Words per Document")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 2)
        plt.hist(char_lengths, bins=30, color='salmon', edgecolor='black')
        plt.title(f"Histogram of Character Counts ({split})")
        plt.xlabel("Characters per Document")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        label_counts = Counter(y)
        classes = [self.class_names[i] for i in label_counts.keys()]
        values = list(label_counts.values())
        plt.bar(classes, values, color='orange')
        plt.title(f"Class Distribution ({split})")
        plt.xticks(rotation=90)
        plt.ylabel("Samples")
        plt.tight_layout()
        plt.show()
