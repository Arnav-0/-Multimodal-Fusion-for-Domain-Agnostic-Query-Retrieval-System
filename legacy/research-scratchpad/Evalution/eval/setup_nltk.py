#!/usr/bin/env python3
"""
Download necessary NLTK data for METEOR and BLEU evaluation metrics
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK datasets"""
    required_data = [
        'punkt',        # For tokenization
        'wordnet',      # For METEOR semantic matching
        'omw-1.4',      # Open Multilingual Wordnet
        'punkt_tab'     # Additional punkt data
    ]
    
    print("Downloading NLTK data for enhanced metrics...")
    
    for data_name in required_data:
        try:
            print(f"Downloading {data_name}...")
            nltk.download(data_name, quiet=False)
            print(f"✅ {data_name} downloaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to download {data_name}: {e}")
    
    print("\n✅ NLTK data download completed!")
    
    # Test imports to verify everything works
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.translate.bleu_score import sentence_bleu
        from rouge_score import rouge_scorer
        print("✅ All metric libraries are working correctly!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)