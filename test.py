#!/usr/bin/env python
"""
Convenience wrapper to run dataset validation tests from root directory.
Delegates to processing_code.test_dataset.
"""

import sys
from pathlib import Path

# Add processing_code to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent / 'processing_code'))

if __name__ == '__main__':
    from test_dataset import main
    main()
