"""
Test script to validate the HistoricalTextDataset.

Shows example training samples and verifies masking is applied correctly.
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer
import sys

# Import the dataset class
sys.path.insert(0, str(Path(__file__).parent))
from pretrain_bert import HistoricalTextDataset


def decode_sample(sample, tokenizer):
    """Decode a training sample for visualization."""
    input_ids = sample['input_ids']
    labels = sample['labels']
    attention_mask = sample['attention_mask']
    
    # Decode input
    tokens = tokenizer.decode(input_ids)
    
    # Find which tokens were masked
    masked_positions = []
    for i in range(len(labels)):
        if labels[i] != -100 and input_ids[i] == tokenizer.mask_token_id:
            original_token = tokenizer.decode([labels[i]])
            masked_positions.append((i, original_token))
    
    return {
        'decoded_input': tokens,
        'masked_positions': masked_positions,
        'attention_mask': attention_mask.sum().item(),
    }


def main():
    """Test the dataset on a sample file."""
    print("=" * 80)
    print("BERT Pretraining Dataset Validation")
    print("=" * 80)
    
    # Check if data exists
    data_dir = Path('data')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"\nError: No CSV files found in {data_dir}")
        print("Please ensure CSV files are in the data/ directory")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    # Load tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = HistoricalTextDataset(
        csv_paths=csv_files,
        tokenizer=tokenizer,
        max_chunk_length=250,
        max_samples=100,  # Just 100 for testing
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("Dataset is empty. Check CSV files for data.")
        return
    
    # Show sample statistics
    print("\n" + "=" * 80)
    print("Sample Statistics")
    print("=" * 80)
    
    total_tokens = 0
    masked_count = 0
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        total_tokens += sample['attention_mask'].sum().item()
        masked_count += (sample['labels'] != -100).sum().item()
    
    avg_tokens = total_tokens / min(5, len(dataset))
    avg_masked = masked_count / min(5, len(dataset))
    
    print(f"Average tokens per sample: {avg_tokens:.1f}")
    print(f"Average masked tokens per sample: {avg_masked:.1f}")
    print(f"Average mask ratio: {avg_masked/avg_tokens*100:.1f}%")
    
    # Show example samples
    print("\n" + "=" * 80)
    print("Example Training Samples")
    print("=" * 80)
    
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i+1} ---")
        sample = dataset[i]
        
        # Decode input
        input_text = tokenizer.decode(sample['input_ids'].tolist(), skip_special_tokens=False)
        print(f"\nInput (truncated): {input_text[:200]}...")
        
        # Show attention
        actual_length = sample['attention_mask'].sum().item()
        print(f"Token length: {actual_length} (max 512)")
        
        # Show masked tokens
        masked_indices = (sample['labels'] != -100).nonzero(as_tuple=True)[0]
        if len(masked_indices) > 0:
            print(f"\nMasked positions: {len(masked_indices)}")
            for idx in masked_indices[:5]:
                idx = idx.item()
                original_id = sample['labels'][idx].item()
                original_token = tokenizer.decode([original_id])
                current_token = tokenizer.decode([sample['input_ids'][idx].item()])
                print(f"  Position {idx}: {current_token} (original: {original_token})")
            if len(masked_indices) > 5:
                print(f"  ... and {len(masked_indices) - 5} more")
    
    # Check special token
    print("\n" + "=" * 80)
    print("Special Tokens")
    print("=" * 80)
    print(f"[TIME] token present: {'[TIME]' in tokenizer.vocab}")
    if '[TIME]' in tokenizer.vocab:
        print(f"[TIME] token ID: {tokenizer.vocab['[TIME]']}")
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)
    print("\nReady to train. Run:")
    print("  python pretrain_bert.py --data_dir data --epochs 3")


if __name__ == '__main__':
    main()
