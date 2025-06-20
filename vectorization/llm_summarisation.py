"""
LLM Summarization Module using Ollama with Gemma 3B

This module provides functionality to generate concise summaries (3-4 sentences)
from text data using Ollama with the Gemma 3B model.
"""

import requests
import time
from typing import Union, List
import pandas as pd


class OllamaSummarizer:
    """
    A minimal class to generate summaries using Ollama with Gemma 3B model.
    """
    
    def __init__(self, model_name: str = "gemma:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama Summarizer.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
    
    def summarize(self, text: str, max_sentences: int = 4, max_length: int = 150, 
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a summary for a single text.
        
        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences in the summary
            max_length: Maximum length of the summary
            temperature: Temperature for text generation
            top_p: Top-p sampling parameter
            
        Returns:
            Generated summary
        """
        if not text or not text.strip():
            return ""
        
        prompt = f"""Please provide a concise summary of the following text in exactly {max_sentences} sentences or fewer. Focus on the key points and main ideas:

Text: {text}

Summary:"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_length,
                "stop": ["\n\n", "Text:", "Original:"]
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get('response', '').strip()
                
                # Clean up the summary
                prefixes = ["Summary:", "The summary is:", "Here's the summary:"]
                for prefix in prefixes:
                    if summary.startswith(prefix):
                        summary = summary[len(prefix):].strip()
                
                # Limit sentences
                sentences = summary.split('.')
                sentences = [s.strip() for s in sentences if s.strip()]
                if len(sentences) > max_sentences:
                    sentences = sentences[:max_sentences]
                
                return '. '.join(sentences) + '.'
            else:
                return self._fallback_summary(text, max_sentences)
                
        except Exception:
            return self._fallback_summary(text, max_sentences)
    
    def _fallback_summary(self, text: str, max_sentences: int) -> str:
        """Generate a simple fallback summary."""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        limited_sentences = sentences[:max_sentences]
        return '. '.join(limited_sentences) + '.'
    
    def summarize_batch(self, texts: Union[List[str], pd.Series], 
                       max_sentences: int = 4, max_length: int = 150,
                       temperature: float = 0.7, top_p: float = 0.9,
                       batch_size: int = 32) -> List[str]:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts: List of texts to summarize
            max_sentences: Maximum number of sentences in each summary
            max_length: Maximum length of each summary
            temperature: Temperature for text generation
            top_p: Top-p sampling parameter
            batch_size: Batch size for processing (not used in current implementation)
            
        Returns:
            List of generated summaries
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        summaries = []
        for text in texts:
            summary = self.summarize(text, max_sentences, max_length, temperature, top_p)
            summaries.append(summary)
            time.sleep(0.1)  # Small delay between requests
        
        return summaries


def create_summaries(texts: Union[List[str], pd.Series],
                    model_name: str = "gemma:3b",
                    max_sentences: int = 4,
                    max_length: int = 150,
                    temperature: float = 0.7,
                    top_p: float = 0.9,
                    batch_size: int = 32) -> List[str]:
    """
    Create summaries for a list of texts using Ollama with Gemma 3B.
    
    Args:
        texts: List of texts to summarize
        model_name: Name of the Ollama model to use
        max_sentences: Maximum number of sentences in each summary
        max_length: Maximum length of each summary
        temperature: Temperature for text generation
        top_p: Top-p sampling parameter
        batch_size: Batch size for processing
        
    Returns:
        List of generated summaries
    """
    summarizer = OllamaSummarizer(model_name=model_name)
    return summarizer.summarize_batch(texts, max_sentences, max_length, temperature, top_p, batch_size)


if __name__ == "__main__":
    pass
