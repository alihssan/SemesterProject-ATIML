"""
Script to run data quality analysis and provide cleaning recommendations
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data_quality_analysis import DataQualityAnalyzer

def main():
    """
    Run complete data quality analysis and provide recommendations
    """
    print("🔍 DATA QUALITY ANALYSIS")
    print("="*60)
    print("Analyzing 20 Newsgroups and AG News datasets for cleaning needs...")
    
    try:
        # Run the analysis
        analyzer = DataQualityAnalyzer()
        results = analyzer.run_complete_analysis()
        
        # Provide clear recommendations
        print("\n" + "="*60)
        print("📋 CLEANING RECOMMENDATIONS")
        print("="*60)
        
        ng_quality = results['20_newsgroups_quality']
        ag_quality = results['ag_news_quality']
        
        print("\n🎯 OVERALL ASSESSMENT:")
        
        # Check if cleaning is needed
        ng_issues = sum([ng_quality.get(f'{key}_percentage', 0) > 5 for key in 
                        ['empty_texts', 'very_short_texts', 'texts_with_html', 'texts_with_urls']])
        ag_issues = sum([ag_quality.get(f'{key}_percentage', 0) > 5 for key in 
                        ['empty_texts', 'very_short_texts', 'texts_with_html', 'texts_with_urls']])
        
        if ng_issues > 0 or ag_issues > 0:
            print("⚠️  CLEANING IS RECOMMENDED")
            print("   Both datasets have quality issues that should be addressed")
        else:
            print("✅ MINIMAL CLEANING NEEDED")
            print("   Datasets are generally clean, but some normalization may help")
        
        print("\n📊 SPECIFIC FINDINGS:")
        
        # 20 Newsgroups findings
        print(f"\n📰 20 Newsgroups Dataset:")
        print(f"   - Total samples: {ng_quality['total_samples']:,}")
        print(f"   - Average text length: {ng_quality['avg_text_length']:.1f} words")
        print(f"   - Empty texts: {ng_quality['empty_texts']} ({ng_quality.get('empty_texts_percentage', 0):.2f}%)")
        print(f"   - Very short texts: {ng_quality['very_short_texts']} ({ng_quality.get('very_short_texts_percentage', 0):.2f}%)")
        print(f"   - Texts with HTML: {ng_quality['texts_with_html']} ({ng_quality.get('texts_with_html_percentage', 0):.2f}%)")
        print(f"   - Texts with URLs: {ng_quality['texts_with_urls']} ({ng_quality.get('texts_with_urls_percentage', 0):.2f}%)")
        
        # AG News findings
        print(f"\n📰 AG News Dataset:")
        print(f"   - Total samples: {ag_quality['total_samples']:,}")
        print(f"   - Average text length: {ag_quality['avg_text_length']:.1f} words")
        print(f"   - Empty texts: {ag_quality['empty_texts']} ({ag_quality.get('empty_texts_percentage', 0):.2f}%)")
        print(f"   - Very short texts: {ag_quality['very_short_texts']} ({ag_quality.get('very_short_texts_percentage', 0):.2f}%)")
        print(f"   - Texts with HTML: {ag_quality['texts_with_html']} ({ag_quality.get('texts_with_html_percentage', 0):.2f}%)")
        print(f"   - Texts with URLs: {ag_quality['texts_with_urls']} ({ag_quality.get('texts_with_urls_percentage', 0):.2f}%)")
        
        print("\n🔧 RECOMMENDED CLEANING STEPS:")
        
        # Priority 1: Critical issues
        print("\n🚨 HIGH PRIORITY (Critical Issues):")
        if ng_quality['empty_texts'] > 0 or ag_quality['empty_texts'] > 0:
            print("   1. Remove empty texts")
        if ng_quality.get('texts_with_html_percentage', 0) > 5 or ag_quality.get('texts_with_html_percentage', 0) > 5:
            print("   2. Remove HTML tags")
        
        # Priority 2: Important issues
        print("\n⚠️  MEDIUM PRIORITY (Important Issues):")
        if ng_quality.get('texts_with_urls_percentage', 0) > 10 or ag_quality.get('texts_with_urls_percentage', 0) > 10:
            print("   1. Remove or clean URLs")
        if ng_quality.get('very_short_texts_percentage', 0) > 5 or ag_quality.get('very_short_texts_percentage', 0) > 5:
            print("   2. Filter very short texts (< 10 words)")
        if ng_quality.get('texts_with_extra_spaces_percentage', 0) > 20 or ag_quality.get('texts_with_extra_spaces_percentage', 0) > 20:
            print("   3. Normalize whitespace")
        
        # Priority 3: Normalization
        print("\n📏 LOW PRIORITY (Normalization):")
        print("   1. Convert to lowercase")
        print("   2. Normalize punctuation")
        print("   3. Handle Unicode characters")
        print("   4. Standardize quotes and apostrophes")
        
        print("\n" + "="*60)
        print("💡 RECOMMENDATION SUMMARY")
        print("="*60)
        
        if ng_issues > 0 or ag_issues > 0:
            print("✅ YES, cleaning and normalization are recommended")
            print("\nReasons:")
            print("   - Both datasets contain HTML tags and URLs")
            print("   - Some texts may have inconsistent formatting")
            print("   - Normalization will improve model performance")
            print("   - Cleaning will reduce noise in the data")
        else:
            print("✅ MINIMAL cleaning needed, but normalization is beneficial")
            print("\nReasons:")
            print("   - Datasets are generally clean")
            print("   - Normalization will improve consistency")
            print("   - Standardization helps with model training")
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Implement the recommended cleaning pipeline")
        print("   2. Test cleaning on a small sample first")
        print("   3. Compare model performance with/without cleaning")
        print("   4. Document the cleaning process for reproducibility")
        
        print(f"\n📁 Detailed results saved in: {analyzer.save_dir}/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 