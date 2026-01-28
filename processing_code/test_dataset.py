"""
Test script to validate the BERT pretraining pipeline.

Shows example training samples and verifies the data processing works correctly.
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import sys

# Import the pretraining module
from pretrain_bert import load_csv_as_dataset, tokenize_and_chunk_function


def main():
    """Test the pretraining pipeline on sample files."""
    print("=" * 80)
    print("BERT Pretraining Pipeline Validation")
    print("=" * 80)
    
    # Check if data exists
    data_dir = Path(__file__).parent.parent / 'data'
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
    
    # Add [TIME] token if not present
    if "[TIME]" not in tokenizer.vocab:
        tokenizer.add_tokens(["[TIME]"])
        print("Added [TIME] token to vocabulary")
    
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Load dataset
    print("\nLoading CSV files as Hugging Face Dataset...")
    dataset = load_csv_as_dataset(csv_files)
    print(f"Dataset rows: {len(dataset)}")
    
    # Show sample data
    print("\nFirst row sample:")
    sample_row = dataset[0]
    for key in sample_row:
        value = sample_row[key]
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    
    # Apply tokenization and chunking (on limited samples for testing)
    print("\nApplying tokenization and chunking (first 100 samples)...")
    test_dataset = dataset.select(range(min(100, len(dataset))))
    
    tokenized_dataset = test_dataset.map(
        lambda examples: tokenize_and_chunk_function(
            examples, 
            tokenizer, 
            max_chunk_length=250
        ),
        batched=True,
        batch_size=50,
        remove_columns=['author', 'place', 'date', 'page_text'],
    )
    
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
    
    if len(tokenized_dataset) == 0:
        print("Error: Tokenized dataset is empty. Check CSV data.")
        return
    
    # Show tokenized sample
    print("\nFirst tokenized sample:")
    sample = tokenized_dataset[0]
    print(f"  Input IDs length: {len(sample['input_ids'])}")
    print(f"  Attention mask: {sum(sample['attention_mask'])} active positions")
    print(f"  Decoded: {tokenizer.decode(sample['input_ids'][:50])}...")
    
    # Split train/val
    print("\nSplitting into train/val (80/20)...")
    split_data = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_data['train']
    val_dataset = split_data['test']
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Test data collator
    print("\nTesting data collator (dynamic padding)...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )
    
    # Create a small batch to test
    batch = {
        'input_ids': [sample['input_ids'] for sample in train_dataset[:2]],
        'attention_mask': [sample['attention_mask'] for sample in train_dataset[:2]],
    }
    
    collated = data_collator(batch)
    print(f"  Collated batch shape: {collated['input_ids'].shape}")
    print(f"  Labels shape: {collated['labels'].shape}")
    masked_count = (collated['labels'] != -100).sum().item()
    print(f"  Masked tokens: {masked_count}")
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)
    print("\nReady to train. Run:")
    print("  cd ..")
    print("  python -m processing_code.pretrain_bert --data_dir data --epochs 3")
    print("\nOr from root directory:")
    print("  python pretrain.py --data_dir data --epochs 3")


if __name__ == '__main__':
    main()
