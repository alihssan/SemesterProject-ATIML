#!/usr/bin/env python3
"""
Runner script for AG News Pipeline with Zero-Shot and Few-Shot Learning
"""

import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ag_news_pipeline import AGNewsPipeline

def main():
    """Main function to run the AG News pipeline with command line arguments."""
    
    parser = argparse.ArgumentParser(description='AG News Pipeline with Zero-Shot and Few-Shot Learning')
    
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum samples per class (default: 500)')
    parser.add_argument('--few-shot-shots', nargs='+', type=int, default=[1, 3, 5, 10],
                       help='Number of shots for few-shot learning (default: 1 3 5 10)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--config-mode', type=str, default='ag_news',
                       help='Configuration mode (default: ag_news)')
    parser.add_argument('--skip-zero-shot', action='store_true',
                       help='Skip zero-shot classification')
    parser.add_argument('--skip-few-shot', action='store_true',
                       help='Skip few-shot classification')
    parser.add_argument('--skip-supervised', action='store_true',
                       help='Skip supervised classification')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AG News Pipeline with Zero-Shot and Few-Shot Learning")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max samples per class: {args.max_samples}")
    print(f"Few-shot shots: {args.few_shot_shots}")
    print(f"Random state: {args.random_state}")
    print(f"Config mode: {args.config_mode}")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = AGNewsPipeline(
            config_mode=args.config_mode,
            max_samples_per_class=args.max_samples,
            few_shot_shots=args.few_shot_shots,
            random_state=args.random_state
        )
        
        # Load dataset
        print("\n1. Loading AG News dataset...")
        pipeline.load_dataset()
        
        # Setup zero-shot classifier
        if not args.skip_zero_shot:
            print("\n2. Setting up zero-shot classifier...")
            pipeline.setup_zero_shot_classifier()
            
            print("\n3. Running zero-shot classification...")
            pipeline.run_zero_shot_classification()
        else:
            print("\n2-3. Skipping zero-shot classification...")
        
        # Run few-shot classification
        if not args.skip_few_shot:
            print("\n4. Running few-shot classification...")
            pipeline.run_few_shot_classification()
        else:
            print("\n4. Skipping few-shot classification...")
        
        # Run supervised classification
        if not args.skip_supervised:
            print("\n5. Running supervised classification...")
            pipeline.run_supervised_classification()
        else:
            print("\n5. Skipping supervised classification...")
        
        # Create visualizations
        print("\n6. Creating visualizations...")
        pipeline.create_visualizations()
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Results saved to: {pipeline.results_dir}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 