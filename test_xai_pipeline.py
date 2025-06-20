#!/usr/bin/env python3
"""
Test script for XAI Analysis Pipeline with Ollama + Gemma 3B.
This script tests the pipeline with a small sample to ensure everything works correctly.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xai_analysis_pipeline import XAIAnalysisPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_xai_pipeline():
    """Test the XAI analysis pipeline with a small sample."""
    logger.info("🧪 Testing XAI Analysis Pipeline with Ollama + Gemma 3B")
    
    try:
        # Test with AG News dataset (smaller, faster)
        logger.info("📊 Testing with AG News dataset (5 samples)")
        
        pipeline = XAIAnalysisPipeline(
            dataset_type='ag_news',
            num_samples=5,  # Small sample for testing
            random_state=42
        )
        
        # Run the pipeline
        pipeline.run_complete_analysis()
        
        logger.info("✅ XAI Analysis Pipeline test completed successfully!")
        logger.info(f"📁 Results saved to: {pipeline.results_dir}")
        
        # Check if results were generated
        expected_files = [
            'explanations.json',
            'summary_statistics.json',
            'classification_report.json',
            'detailed_explanations.txt'
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(pipeline.results_dir, file_name)
            if os.path.exists(file_path):
                logger.info(f"✅ {file_name} generated successfully")
            else:
                logger.warning(f"⚠️  {file_name} not found")
        
        # Print summary statistics
        import json
        summary_path = os.path.join(pipeline.results_dir, 'summary_statistics.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            logger.info("📈 Summary Statistics:")
            logger.info(f"   - Dataset: {summary.get('dataset_type', 'Unknown')}")
            logger.info(f"   - Total samples: {summary.get('total_samples', 0)}")
            logger.info(f"   - Mean agreement score: {summary.get('mean_agreement_score', 0):.3f}")
            logger.info(f"   - Mean confidence: {summary.get('mean_confidence', 0):.3f}")
            logger.info(f"   - Accuracy: {summary.get('accuracy', 0):.3f}")
            logger.info(f"   - LIME available: {summary.get('lime_available', 0)}")
            logger.info(f"   - LLM available: {summary.get('llm_available', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ XAI Analysis Pipeline test failed: {str(e)}")
        logger.error("Please check the error message above and ensure:")
        logger.error("1. All dependencies are installed: pip install -r requirements.txt")
        logger.error("2. Ollama is running: ollama serve")
        logger.error("3. Gemma 3B is available: ollama pull gemma:3b")
        return False

def test_ollama_connection():
    """Test Ollama connection specifically."""
    logger.info("🔗 Testing Ollama connection...")
    
    try:
        from vectorization.llm_summarisation import OllamaSummarizer
        
        # Test Ollama connection
        summarizer = OllamaSummarizer(model_name="gemma:3b")
        
        # Test with a simple prompt
        test_prompt = "Explain why this text might be classified as 'sports': 'The team won the championship game.'"
        response = summarizer.summarize(test_prompt, max_sentences=2)
        
        logger.info("✅ Ollama connection test successful!")
        logger.info(f"📝 Sample response: {response[:100]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama connection test failed: {str(e)}")
        logger.error("Please ensure Ollama is running with: ollama serve")
        return False

def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("🧪 XAI ANALYSIS PIPELINE TEST")
    logger.info("=" * 60)
    
    # Test 1: Ollama connection
    ollama_ok = test_ollama_connection()
    
    # Test 2: Full pipeline
    pipeline_ok = test_xai_pipeline()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    if ollama_ok:
        logger.info("✅ Ollama + Gemma 3B: Working")
    else:
        logger.info("❌ Ollama + Gemma 3B: Failed")
    
    if pipeline_ok:
        logger.info("✅ XAI Pipeline: Working")
    else:
        logger.info("❌ XAI Pipeline: Failed")
    
    if ollama_ok and pipeline_ok:
        logger.info("🎉 All tests passed! The XAI Analysis Pipeline is ready to use.")
        logger.info("💡 You can now run: python run_xai_analysis.py --dataset both --samples 15")
    else:
        logger.info("⚠️  Some tests failed. Please check the error messages above.")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 