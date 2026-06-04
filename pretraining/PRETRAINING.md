# BERT Pretraining on Historical Text

This script pretrains a BERT model from Hugging Face on historical text data (EEBO, ECCO, EVAN).

## Features

- **Custom Input Format**: Each training sample contains:
  - `[TIME]` special token
  - Publication date
  - Up to 250 tokens of text
  
- **Masked Language Modeling (MLM)**: 
  - 15% of date tokens are masked
  - 15% of text tokens are masked
  - Both date and text contribute to the loss

- **Data Processing**:
  - Loads CSV files with columns: `author`, `place`, `date`, `page_text`
  - Chunks text into 250-token sequences
  - Handles multiple CSV files (EEBO, ECCO, EVAN)

## Installation

```bash
# Install training dependencies
pip install -r requirements_train.txt
```

## Usage

### Basic Training

```bash
python pretrain_bert.py \
  --data_dir data \
  --output_dir output/bert_pretrained \
  --model_name bert-base-uncased \
  --epochs 3 \
  --batch_size 16
```

### With Custom Parameters

```bash
python pretrain_bert.py \
  --data_dir data \
  --output_dir output/bert_historical \
  --model_name bert-base-cased \
  --chunk_length 250 \
  --batch_size 32 \
  --epochs 5 \
  --learning_rate 3e-5 \
  --seed 42
```

### Test Run (Limited Samples)

```bash
python pretrain_bert.py \
  --data_dir data \
  --output_dir output/bert_test \
  --max_samples 1000 \
  --epochs 1 \
  --batch_size 8
```

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Directory containing CSV files |
| `--output_dir` | `output/bert_pretrained` | Directory to save model |
| `--model_name` | `bert-base-uncased` | Hugging Face model to finetune |
| `--chunk_length` | `250` | Tokens per text chunk |
| `--batch_size` | `16` | Training batch size |
| `--epochs` | `3` | Number of training epochs |
| `--learning_rate` | `5e-5` | Learning rate |
| `--max_samples` | `None` | Optional sample limit for testing |
| `--seed` | `42` | Random seed for reproducibility |

## Input Data Format

The script expects CSV files in the `data/` directory with columns:

```
author | place | date | page_text
-------|-------|------|----------
John Donne | London | 1633 | The text content...
```

Supported formats:
- `eebo_pages_full.csv` - Early English Books Online
- `ecco_pages_full.csv` - Eighteenth Century Collections Online
- `evan_pages_full.csv` - Evans Early American Imprints

## Training Samples

Example training sample structure:

```
[CLS] [TIME] 1633 the text of the document continues here ... [SEP]
```

With masking:
- Some date tokens → `[MASK]`
- Some text tokens → `[MASK]`
- Remaining masked tokens → random words or original

## Output

The trained model is saved to the specified `--output_dir`:

```
output/bert_pretrained/
├── pytorch_model.bin       # Model weights
├── config.json             # Model configuration
├── vocab.txt               # Tokenizer vocabulary
├── tokenizer_config.json   # Tokenizer configuration
└── training_args.bin       # Training arguments
```

## Loading Trained Model

```python
from transformers import BertForMaskedLM, BertTokenizer

model = BertForMaskedLM.from_pretrained('output/bert_pretrained')
tokenizer = BertTokenizer.from_pretrained('output/bert_pretrained')

# Use for downstream tasks or inference
```

## Notes

- The script adds a custom `[TIME]` token to the vocabulary
- 90% of data is used for training, 10% for validation
- Model checkpoints are saved after each epoch
- Best model (by validation loss) is kept
- Training uses gradient accumulation and warmup

## GPU Usage

To use GPU training:

```bash
# Automatic (CUDA if available)
python pretrain_bert.py ...

# Explicitly set device in script if needed
# Modify: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## License

MIT License
