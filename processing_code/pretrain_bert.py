"""
BERT Pretraining Script for Historical Text Data

Loads CSV data (EEBO, ECCO, EVAN) and prepares it for BERT pretraining.
Uses efficient Hugging Face Datasets library with dynamic padding.

Each training sample consists of:
- Date of publication with [TIME] special token
- 250-token text chunk
- Both date and text are masked for MLM training

Key optimizations:
- Dataset.map() with batched=True for fast preprocessing
- Dynamic padding (only pad to batch max, not dataset max)
- Arrow file storage for memory efficiency
- DataCollatorWithPadding for on-the-fly padding
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, BertForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_csv_as_dataset(csv_paths: List[Path]) -> Dataset:
    """
    Load CSV files and convert to Hugging Face Dataset format.
    
    Uses Apache Arrow storage for memory efficiency.
    """
    all_data = []
    
    for csv_path in csv_paths:
        logger.info(f"Loading {csv_path.name}...")
        df = pd.read_csv(csv_path)
        all_data.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined_df)}")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(combined_df)
    return dataset


def chunk_text_function(example, max_chunk_length: int = 250):
    """
    Chunk long text into fixed-length sequences.
    
    This function is applied with map(batched=True) for efficiency.
    """
    # This will be called with batches of examples
    # We need to handle each text individually and potentially create multiple samples
    
    # For now, keep as single row - chunking happens during tokenization
    return example


def tokenize_and_chunk_function(examples, tokenizer, max_chunk_length: int = 250):
    """
    Tokenize text and create chunks with date prefix.
    
    Applied with map(batched=True) for efficient preprocessing.
    Uses the fast Rust-backed tokenizer.
    
    Format: [CLS] <date> [TIME] <text_chunk> [SEP]
    """
    batch_size = len(examples['date'])
    all_input_ids = []
    all_attention_masks = []
    
    for idx in range(batch_size):
        date = examples['date'][idx]
        text = examples['page_text'][idx]
        
        if not text or pd.isna(text) or pd.isna(date):
            continue
        
        date_str = str(date).strip()
        text_str = str(text).strip()
        
        # Tokenize date
        date_tokens = tokenizer.tokenize(date_str)
        
        # Tokenize text
        text_tokens = tokenizer.tokenize(text_str)
        
        # Create chunks of text with date prefix
        # Each chunk: [CLS] <date> [TIME] <text_chunk> [SEP]
        for chunk_start in range(0, len(text_tokens), max_chunk_length):
            chunk_end = min(chunk_start + max_chunk_length, len(text_tokens))
            chunk_tokens = text_tokens[chunk_start:chunk_end]
            
            # Build full sequence
            tokens = [tokenizer.cls_token]
            tokens.extend(date_tokens)
            tokens.append("[TIME]")
            tokens.extend(chunk_tokens)
            tokens.append(tokenizer.sep_token)
            
            # Convert to IDs (no padding yet - dynamic padding happens in collator)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Check if within model limits
            if len(input_ids) <= 512:  # BERT max length
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
    
    if not all_input_ids:
        # Return empty batch if no valid samples
        return {
            'input_ids': [],
            'attention_mask': [],
        }
    
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
    }


def main():
    parser = argparse.ArgumentParser(description='Pretrain BERT on historical text data')
    parser.add_argument('--data_dir', type=Path, default=Path('data'),
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=Path, default=Path('output/bert_pretrained'),
                        help='Directory to save model')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Hugging Face model name')
    parser.add_argument('--chunk_length', type=int, default=250,
                        help='Token chunk length for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Find CSV files
    csv_files = list(args.data_dir.glob('*.csv'))
    if not csv_files:
        logger.error(f"No CSV files found in {args.data_dir}")
        return
    
    logger.info(f"Found CSV files: {[f.name for f in csv_files]}")
    
    # Load dataset
    logger.info("Loading datasets from CSV files...")
    dataset = load_csv_as_dataset(csv_files)
    
    # Add special token if needed
    if "[TIME]" not in tokenizer.vocab:
        tokenizer.add_tokens(["[TIME]"])
        logger.info("Added [TIME] token to tokenizer")
    
    # Tokenize and chunk using map() with batched=True (efficient!)
    logger.info("Tokenizing and chunking text (this may take a while)...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_chunk_function(
            examples, 
            tokenizer, 
            max_chunk_length=args.chunk_length
        ),
        batched=True,
        batch_size=1000,  # Process 1000 examples at a time
        remove_columns=['author', 'place', 'page_text'],  # Remove original columns
        num_proc=4,  # Use 4 processes for faster preprocessing
    )
    
    logger.info(f"Preprocessed dataset size: {len(tokenized_dataset)}")
    
    # Split into train/val (90/10)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = BertForMaskedLM.from_pretrained(args.model_name)
    
    # Resize token embeddings for new [TIME] token
    model.resize_token_embeddings(len(tokenizer))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Data collator with dynamic padding (crucial for efficiency!)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
