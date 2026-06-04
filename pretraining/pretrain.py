#!/usr/bin/env python
"""
Convenience wrapper to run BERT pretraining from root directory.
Delegates to processing_code.pretrain_bert.
"""

import sys
from pathlib import Path

# Add processing_code to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent / 'processing_code'))

if __name__ == '__main__':
    from pretrain_bert import main
    main()
