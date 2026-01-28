"""
BERT Pretraining Script for Historical Text Data

Loads CSV data (EEBO, ECCO, EVAN) and prepares it for BERT pretraining.
Each training sample consists of:
- Date of publication with [TIME] special token
- 250-token text chunk
- Both date and text are masked for MLM training
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, TextDatasetForNextSentencePrediction
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalTextDataset(Dataset):
    """
    Dataset for pretraining BERT on historical text data.
    
    Each sample:
    - Starts with [TIME] <date> tokens
    - Followed by up to 250 tokens of text
    - Both date and text are masked for MLM
    """
    
    def __init__(
        self,
        csv_paths: List[Path],
        tokenizer,
        max_chunk_length: int = 250,
        date_mask_prob: float = 0.15,
        text_mask_prob: float = 0.15,
        max_samples: int = None,
    ):
        """
        Args:
            csv_paths: List of CSV file paths to load
            tokenizer: BERT tokenizer
            max_chunk_length: Maximum tokens per text chunk (default 250)
            date_mask_prob: Probability of masking date tokens
            text_mask_prob: Probability of masking text tokens
            max_samples: Optional limit on number of samples
        """
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length
        self.date_mask_prob = date_mask_prob
        self.text_mask_prob = text_mask_prob
        self.samples = []
        
        # Add special token for time if not already present
        if "[TIME]" not in tokenizer.vocab:
            tokenizer.add_tokens(["[TIME]"])
        
        self._load_data(csv_paths, max_samples)
    
    def _load_data(self, csv_paths: List[Path], max_samples: int = None):
        """Load and chunk data from CSV files."""
        total_samples = 0
        
        for csv_path in csv_paths:
            logger.info(f"Loading data from {csv_path.name}")
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                if max_samples and total_samples >= max_samples:
                    logger.info(f"Reached max samples limit: {max_samples}")
                    return
                
                date = str(row['date']).strip()
                text = str(row['page_text']).strip()
                
                if not text:
                    continue
                
                # Tokenize text
                text_tokens = self.tokenizer.tokenize(text)
                
                # Create chunks of max_chunk_length tokens
                for i in range(0, len(text_tokens), self.max_chunk_length):
                    if max_samples and total_samples >= max_samples:
                        return
                    
                    chunk_tokens = text_tokens[i:i + self.max_chunk_length]
                    
                    # Create sample: [CLS] [TIME] <date> <text> [SEP]
                    self.samples.append({
                        'date': date,
                        'text_tokens': chunk_tokens,
                        'author': row.get('author', 'Unknown'),
                        'place': row.get('place', 'Unknown'),
                    })
                    
                    total_samples += 1
        
        logger.info(f"Total samples created: {len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample with masking applied."""
        sample = self.samples[idx]
        date = sample['date']
        text_tokens = sample['text_tokens']
        
        # Tokenize date
        date_tokens = self.tokenizer.tokenize(date)
        
        # Build token sequence: [CLS] [TIME] <date> <text> [SEP]
        tokens = [self.tokenizer.cls_token]
        tokens.append("[TIME]")
        tokens.extend(date_tokens)
        tokens.extend(text_tokens)
        tokens.append(self.tokenizer.sep_token)
        
        # Ensure we don't exceed model's max length
        max_length = 512  # Standard BERT max length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        # Create labels for MLM (copy of input_ids, -100 for non-masked tokens)
        labels = input_ids.copy()
        
        # Mask date tokens ([TIME] and date tokens)
        date_end_idx = 2 + len(date_tokens)  # [CLS] [TIME] <date>
        for i in range(1, date_end_idx):  # Skip [CLS]
            if np.random.random() < self.date_mask_prob:
                if np.random.random() < 0.8:
                    input_ids[i] = self.tokenizer.mask_token_id
                elif np.random.random() < 0.5:
                    input_ids[i] = np.random.randint(0, len(self.tokenizer))
                # else: keep original token
            else:
                labels[i] = -100  # Not masked, so loss is 0
        
        # Mask text tokens
        for i in range(date_end_idx, len(input_ids) - 1):  # Skip [SEP] and padding
            if input_ids[i] == self.tokenizer.pad_token_id:
                labels[i] = -100  # Don't compute loss on padding
            elif np.random.random() < self.text_mask_prob:
                labels[i] = input_ids[i]  # Store original for loss computation
                if np.random.random() < 0.8:
                    input_ids[i] = self.tokenizer.mask_token_id
                elif np.random.random() < 0.5:
                    input_ids[i] = np.random.randint(0, len(self.tokenizer))
                # else: keep original token
            else:
                labels[i] = -100  # Not masked, so loss is 0
        
        # Set [CLS] and [SEP] labels to -100 (don't compute loss on these)
        labels[0] = -100
        labels[len(input_ids) - 1 - padding_length] = -100
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
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
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = HistoricalTextDataset(
        csv_paths=csv_files,
        tokenizer=tokenizer,
        max_chunk_length=args.chunk_length,
        max_samples=args.max_samples,
    )
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
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
    
    # Data collator
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
