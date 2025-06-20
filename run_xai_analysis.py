#!/usr/bin/env python3
"""
Run XAI Analysis Pipeline for both 20 Newsgroups and AG News datasets.
Uses LIME for explainability and Ollama + Gemma 3B for LLM reasoning.
"""

import os
import sys
import argparse
import logging
import subprocess
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

def check_ollama_setup():
    """Check if Ollama is properly set up with Gemma 3B."""
    logger.info("Checking Ollama setup...")
    
    try:
        # Check if Ollama is running
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.error("❌ Ollama is not running or not installed")
            logger.info("Please install Ollama from: https://ollama.ai/")
            logger.info("Then start it with: ollama serve")
            return False
        
        # Check if Gemma 3B is available
        if 'gemma:3b' in result.stdout:
            logger.info("✅ Ollama is running and Gemma 3B is available")
            return True
        else:
            logger.warning("⚠️  Ollama is running but Gemma 3B is not available")
            logger.info("Please pull Gemma 3B with: ollama pull gemma:3b")
            return False
            
    except FileNotFoundError:
        logger.error("❌ Ollama is not installed")
        logger.info("Please install Ollama from: https://ollama.ai/")
        return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Ollama is not responding")
        logger.info("Please start Ollama with: ollama serve")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking Ollama: {e}")
        return False

def run_xai_analysis(dataset_type: str, num_samples: int = 15, random_state: int = 42):
    """
    Run XAI analysis for the specified dataset.
    
    Args:
        dataset_type: 'newsgroups' or 'ag_news'
        num_samples: Number of samples to analyze
        random_state: Random seed
    """
    logger.info(f"Starting XAI analysis for {dataset_type} dataset...")
    
    try:
        # Create and run pipeline
        pipeline = XAIAnalysisPipeline(
            dataset_type=dataset_type,
            num_samples=num_samples,
            random_state=random_state
        )
        
        pipeline.run_complete_analysis()
        
        logger.info(f"✅ XAI analysis completed for {dataset_type} dataset!")
        logger.info(f"📁 Results saved to: {pipeline.results_dir}")
        
        return pipeline.results_dir
        
    except Exception as e:
        logger.error(f"❌ XAI analysis failed for {dataset_type}: {str(e)}")
        raise

def main():
    """Main function to run XAI analysis for both datasets."""
    parser = argparse.ArgumentParser(description='Run XAI Analysis Pipeline with Ollama + Gemma 3B')
    parser.add_argument('--dataset', choices=['newsgroups', 'ag_news', 'both'], 
                       default='both', help='Dataset to analyze')
    parser.add_argument('--samples', type=int, default=15, 
                       help='Number of samples to analyze per dataset')
    parser.add_argument('--random_state', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory for results')
    parser.add_argument('--skip_ollama_check', action='store_true',
                       help='Skip Ollama setup check')
    
    args = parser.parse_args()
    
    # Check Ollama setup unless skipped
    if not args.skip_ollama_check:
        if not check_ollama_setup():
            logger.warning("⚠️  Continuing without Ollama - LLM explanations will be disabled")
            logger.info("To enable LLM explanations, please set up Ollama with Gemma 3B")
    
    # Create base output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"results/xai_analysis_comparison_{timestamp}"
    else:
        base_output_dir = args.output_dir
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    results = {}
    
    if args.dataset in ['newsgroups', 'both']:
        logger.info("=" * 60)
        logger.info("🔍 ANALYZING 20 NEWSGROUPS DATASET")
        logger.info("=" * 60)
        
        try:
            results_dir = run_xai_analysis(
                dataset_type='newsgroups',
                num_samples=args.samples,
                random_state=args.random_state
            )
            results['newsgroups'] = results_dir
        except Exception as e:
            logger.error(f"❌ Failed to analyze newsgroups dataset: {e}")
            results['newsgroups'] = None
    
    if args.dataset in ['ag_news', 'both']:
        logger.info("=" * 60)
        logger.info("🔍 ANALYZING AG NEWS DATASET")
        logger.info("=" * 60)
        
        try:
            results_dir = run_xai_analysis(
                dataset_type='ag_news',
                num_samples=args.samples,
                random_state=args.random_state
            )
            results['ag_news'] = results_dir
        except Exception as e:
            logger.error(f"❌ Failed to analyze AG news dataset: {e}")
            results['ag_news'] = None
    
    # Print summary
    logger.info("=" * 60)
    logger.info("📊 XAI ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    successful_analyses = 0
    for dataset, result_dir in results.items():
        if result_dir:
            logger.info(f"✅ {dataset.upper()}: Results saved to {result_dir}")
            successful_analyses += 1
        else:
            logger.info(f"❌ {dataset.upper()}: Analysis failed")
    
    logger.info("=" * 60)
    if successful_analyses > 0:
        logger.info(f"🎉 XAI Analysis Pipeline completed! {successful_analyses} dataset(s) analyzed successfully.")
    else:
        logger.error("💥 XAI Analysis Pipeline failed for all datasets.")
    logger.info("=" * 60)
    
    # Print next steps
    if successful_analyses > 0:
        logger.info("📋 Next Steps:")
        logger.info("1. Check the results directories for detailed explanations")
        logger.info("2. Review the visualizations (PNG files)")
        logger.info("3. Read detailed_explanations.txt for sample analyses")
        logger.info("4. Examine summary_statistics.json for overall metrics")

if __name__ == "__main__":
    main() 