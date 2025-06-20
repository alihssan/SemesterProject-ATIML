"""
Pipeline Runner Script

This script provides an easy way to run the complete 20 Newsgroups analysis pipeline
with different configurations.
"""

import argparse
import logging
from main_pipeline import NewsgroupsPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the pipeline with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run 20 Newsgroups Analysis Pipeline')
    
    parser.add_argument('--categories', nargs='+', 
                       default=None,
                       help='Newsgroup categories to use')
    
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per category')
    
    parser.add_argument('--test_size', type=float, default=None,
                       help='Proportion of data for testing')
    
    parser.add_argument('--random_state', type=int, default=None,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--config_mode', type=str, default='full',
                       choices=['full', 'quick'],
                       help='Configuration mode (full or quick)')
    
    parser.add_argument('--quick', action='store_true',
                       help='Run with minimal samples for quick testing (overrides config_mode)')
    
    args = parser.parse_args()
    
    # Set config mode based on quick flag
    config_mode = 'quick' if args.quick else args.config_mode
    
    logger.info("Starting 20 Newsgroups Analysis Pipeline")
    logger.info(f"Configuration mode: {config_mode}")
    logger.info(f"Categories: {args.categories}")
    logger.info(f"Max samples per category: {args.max_samples}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random state: {args.random_state}")
    
    try:
        # Initialize and run pipeline
        pipeline = NewsgroupsPipeline(
            categories=args.categories,
            max_samples_per_category=args.max_samples,
            test_size=args.test_size,
            random_state=args.random_state,
            config_mode=config_mode
        )
        
        # Run complete pipeline
        pipeline.run_complete_pipeline()
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {pipeline.results_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 