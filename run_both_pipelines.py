#!/usr/bin/env python3
"""
Run Both Pipelines Sequentially

This script runs:
1. Main Pipeline (20 Newsgroups) - complete analysis with TF-IDF, Doc2Vec, Sentence-BERT
2. AG News Pipeline - zero-shot and few-shot learning analysis

Both pipelines will be executed in sequence, and results will be saved to separate directories.
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('both_pipelines.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_main_pipeline(args):
    """Run the main pipeline (20 Newsgroups)."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING MAIN PIPELINE (20 Newsgroups)")
    logger.info("=" * 60)
    
    try:
        # Import main pipeline module without executing main()
        import main_pipeline
        
        # Create pipeline with specified parameters
        pipeline = main_pipeline.NewsgroupsPipeline(
            categories=args.categories,
            max_samples_per_category=args.max_samples,
            test_size=args.test_size,
            random_state=args.random_state,
            config_mode=args.config_mode
        )
        
        # Run the complete pipeline
        pipeline.run_complete_pipeline()
        
        logger.info("✅ Main pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {pipeline.results_dir}")
        
        return pipeline.results_dir
        
    except Exception as e:
        logger.error(f"❌ Main pipeline failed: {str(e)}")
        raise

def run_ag_news_pipeline(args):
    """Run the AG News pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING AG NEWS PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Import AG news pipeline module without executing main()
        import ag_news_pipeline
        
        # Check if required dependencies are available
        try:
            import transformers
            import torch
            import datasets
            logger.info("✅ Transformers, PyTorch, and Datasets dependencies available")
        except ImportError as e:
            logger.error(f"❌ Missing required dependencies for AG News pipeline: {e}")
            logger.error("Please install transformers, torch, and datasets: pip install transformers torch datasets")
            raise ImportError(f"AG News pipeline requires transformers, torch, and datasets: {e}")
        
        # Create pipeline with specified parameters
        pipeline = ag_news_pipeline.AGNewsPipeline(
            config_mode=args.ag_config_mode,
            max_samples_per_class=args.ag_max_samples,
            few_shot_shots=args.few_shot_shots,
            random_state=args.random_state
        )
        
        # Run the complete pipeline
        pipeline.run_complete_pipeline()
        
        logger.info("✅ AG News pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {pipeline.results_dir}")
        
        return pipeline.results_dir
        
    except ImportError as e:
        logger.error(f"❌ AG News pipeline failed due to missing dependencies: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ AG News pipeline failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def run_xai_analysis(args):
    """Run XAI analysis on both datasets if requested."""
    if not args.run_xai:
        logger.info("⏭️  Skipping XAI analysis (--run_xai not specified)")
        return None
    
    logger.info("=" * 60)
    logger.info("🔍 STARTING XAI ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Import and run XAI analysis
        from xai_analysis_pipeline import XAIAnalysisPipeline
        
        results = {}
        
        # Run XAI analysis for newsgroups
        if args.xai_datasets in ['newsgroups', 'both']:
            logger.info("📊 Running XAI analysis for 20 Newsgroups...")
            newsgroups_xai = XAIAnalysisPipeline(
                dataset_type='newsgroups',
                num_samples=args.xai_samples,
                random_state=args.random_state
            )
            newsgroups_xai.run_complete_analysis()
            results['newsgroups'] = newsgroups_xai.results_dir
        
        # Run XAI analysis for AG news
        if args.xai_datasets in ['ag_news', 'both']:
            logger.info("📊 Running XAI analysis for AG News...")
            ag_news_xai = XAIAnalysisPipeline(
                dataset_type='ag_news',
                num_samples=args.xai_samples,
                random_state=args.random_state
            )
            ag_news_xai.run_complete_analysis()
            results['ag_news'] = ag_news_xai.results_dir
        
        logger.info("✅ XAI analysis completed successfully!")
        for dataset, result_dir in results.items():
            logger.info(f"📁 {dataset.upper()} XAI results: {result_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ XAI analysis failed: {str(e)}")
        raise

def main():
    """Main function to run both pipelines."""
    parser = argparse.ArgumentParser(description='Run both main pipeline and AG news pipeline')
    
    # Main pipeline arguments
    parser.add_argument('--categories', nargs='+', default=None,
                       help='Newsgroup categories to use (default: all)')
    parser.add_argument('--max_samples', type=int, default=200,
                       help='Maximum samples per category for main pipeline')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--config_mode', choices=['full', 'quick'], default='full',
                       help='Configuration mode for main pipeline')
    
    # AG News pipeline arguments
    parser.add_argument('--ag_max_samples', type=int, default=500,
                       help='Maximum samples per class for AG news pipeline')
    parser.add_argument('--ag_config_mode', default='ag_news',
                       help='Configuration mode for AG news pipeline')
    parser.add_argument('--few_shot_shots', nargs='+', type=int, default=[1, 3, 5, 10],
                       help='Few-shot learning shot counts')
    
    # XAI analysis arguments
    parser.add_argument('--run_xai', action='store_true',
                       help='Run XAI analysis after pipelines')
    parser.add_argument('--xai_datasets', choices=['newsgroups', 'ag_news', 'both'], default='both',
                       help='Datasets to run XAI analysis on')
    parser.add_argument('--xai_samples', type=int, default=15,
                       help='Number of samples for XAI analysis')
    
    # General arguments
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--skip_main', action='store_true',
                       help='Skip main pipeline')
    parser.add_argument('--skip_ag_news', action='store_true',
                       help='Skip AG news pipeline')
    
    args = parser.parse_args()
    
    # Create summary directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = f"results/both_pipelines_{timestamp}"
    os.makedirs(summary_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🎯 RUNNING BOTH PIPELINES SEQUENTIALLY")
    logger.info("=" * 80)
    logger.info(f"📅 Timestamp: {timestamp}")
    logger.info(f"🎲 Random State: {args.random_state}")
    logger.info(f"📁 Summary Directory: {summary_dir}")
    
    results = {}
    
    # Run main pipeline
    if not args.skip_main:
        try:
            main_results = run_main_pipeline(args)
            results['main_pipeline'] = main_results
        except Exception as e:
            logger.error(f"❌ Main pipeline failed: {e}")
            results['main_pipeline'] = None
    else:
        logger.info("⏭️  Skipping main pipeline (--skip_main specified)")
    
    # Run AG news pipeline
    if not args.skip_ag_news:
        try:
            ag_results = run_ag_news_pipeline(args)
            results['ag_news_pipeline'] = ag_results
        except Exception as e:
            logger.error(f"❌ AG News pipeline failed: {e}")
            results['ag_news_pipeline'] = None
    else:
        logger.info("⏭️  Skipping AG news pipeline (--skip_ag_news specified)")
    
    # Run XAI analysis
    if args.run_xai:
        try:
            xai_results = run_xai_analysis(args)
            results['xai_analysis'] = xai_results
        except Exception as e:
            logger.error(f"❌ XAI analysis failed: {e}")
            results['xai_analysis'] = None
    
    # Create summary
    logger.info("=" * 80)
    logger.info("📊 PIPELINES SUMMARY")
    logger.info("=" * 80)
    
    successful_pipelines = 0
    for pipeline_name, result_dir in results.items():
        if result_dir:
            logger.info(f"✅ {pipeline_name.upper()}: {result_dir}")
            successful_pipelines += 1
        else:
            logger.info(f"❌ {pipeline_name.upper()}: Failed")
    
    # Save summary to file
    summary_file = os.path.join(summary_dir, "pipeline_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("BOTH PIPELINES EXECUTION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Random State: {args.random_state}\n\n")
        
        for pipeline_name, result_dir in results.items():
            if result_dir:
                f.write(f"✅ {pipeline_name.upper()}: {result_dir}\n")
            else:
                f.write(f"❌ {pipeline_name.upper()}: Failed\n")
        
        f.write(f"\nSuccessful Pipelines: {successful_pipelines}/{len(results)}\n")
    
    logger.info("=" * 80)
    if successful_pipelines > 0:
        logger.info(f"🎉 Execution completed! {successful_pipelines} pipeline(s) successful.")
        logger.info(f"📋 Summary saved to: {summary_file}")
    else:
        logger.error("💥 All pipelines failed!")
    
    logger.info("=" * 80)
    
    # Print next steps
    if successful_pipelines > 0:
        logger.info("📋 Next Steps:")
        logger.info("1. Check individual pipeline result directories")
        logger.info("2. Review generated visualizations and reports")
        logger.info("3. Examine summary_statistics.json files")
        if args.run_xai:
            logger.info("4. Review XAI explanations and agreement scores")

if __name__ == "__main__":
    main() 