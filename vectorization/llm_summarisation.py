"""
LLM Summarization Module using Ollama with Gemma 3B

This module provides functionality to generate concise summaries (3-4 sentences)
from text data using Ollama with the Gemma 3B model.
"""

import time
from typing import Union, List
import pandas as pd
import ollama
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OllamaSummarizer:
    """
    A minimal class to generate summaries using Ollama with Gemma 3B model.
    """

    def __init__(self, model_name: str = "gemma3:4b"):
        """
        Initialize the Ollama Summarizer.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        logging.info(f"Initialized OllamaSummarizer with model: {self.model_name}")

    def summarize(self, text: str, max_sentences: int = 4) -> str:
        """
        Generate a summary for a single text.

        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences in the summary

        Returns:
            Generated summary
        """
        if not text or not text.strip():
            logging.warning("Empty input text. Returning empty summary.")
            return ""

        prompt = f"""Please provide a concise summary of the following text in exactly {max_sentences} sentences or fewer. Focus on the key points and main ideas:\n\nText: {text}\n\nSummary:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response['message']['content'].strip()

            # Clean up and truncate to max_sentences
            sentences = summary.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > max_sentences:
                sentences = sentences[:max_sentences]

            logging.info(f"Generated summary with {len(sentences)} sentence(s).")
            return '. '.join(sentences) + '.'

        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return self._fallback_summary(text, max_sentences)

    def _fallback_summary(self, text: str, max_sentences: int) -> str:
        logging.warning("Using fallback summary method.")
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= max_sentences:
            return text
        return '. '.join(sentences[:max_sentences]) + '.'

    def summarize_batch(self, texts: Union[List[str], pd.Series], max_sentences: int = 4, batch_size: int = 32) -> List[str]:
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        summaries = []
        total = len(texts)
        logging.info(f"Starting batch summarization for {total} texts...")

        for idx, text in enumerate(texts):
            summary = self.summarize(text, max_sentences)
            summaries.append(summary)
            logging.info(f"[{idx+1}/{total}] Summary generated.")
            time.sleep(0.1)  # Small delay between requests

        logging.info("Batch summarization completed.")
        return summaries


def create_summaries(texts: Union[List[str], pd.Series],
                     model_name: str = "gemma3:4b",
                     max_sentences: int = 4,
                     batch_size: int = 32) -> List[str]:
    logging.info("Creating summaries using OllamaSummarizer...")
    summarizer = OllamaSummarizer(model_name=model_name)
    return summarizer.summarize_batch(texts, max_sentences, batch_size=batch_size)


if __name__ == "__main__":
    pass
