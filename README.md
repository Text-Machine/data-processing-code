# Processing Code Package

Data processing utilities for converting P4 XML files to structured CSV data.

## Installation

From the project root:

```bash
# Install in development mode
pip install -e .

# Or, add to your path when using notebooks
import sys
sys.path.insert(0, 'processing_code')
```

## Usage

### Basic Usage

```python
from processing_code import parse_xml, process_files

# Parse a single file
pages = parse_xml('path/to/file.xml')

# Process multiple files
df = process_files(
    xml_files=['file1.xml', 'file2.xml'],
    output_path='output.csv'
)
```

### Supported Formats

- **EEBO** (Early English Books Online) Phase 1 - P4 XML
- **ECCO** (Eighteenth Century Collections Online) - P4 XML
- **EVAN** (Evans Early American Imprints) - P4 XML

All formats use uppercase element names: `ETS`, `HEADER`, `EEBO`, `TEXT`, `PB`, etc.

### Output Format

CSV file with page-level granularity:

| author | place | date | page_text |
|--------|-------|------|-----------|
| William Shakespeare | London | 1623 | First page of text... |
| William Shakespeare | London | 1623 | Second page of text... |

Each row represents one page from a document.

## Testing

```bash
cd processing_code
python -m pytest test_text_parser.py -v
```

Or:

```bash
python -m unittest test_text_parser
```

## Module Structure

```
processing_code/
├── __init__.py           # Package initialization and exports
├── text_parser.py        # Core parsing logic
├── test_text_parser.py   # Unit tests
└── README.md             # This file
```

## Functions

### `parse_xml(xml_path) -> List[Dict]`

Parse a single P4 XML file and extract metadata and page-level text.

**Args:**
- `xml_path`: Path to P4 XML file

**Returns:** List of dicts with keys: `author`, `place`, `date`, `page_text`

### `process_files(xml_files, output_path=None, max_files=None) -> DataFrame`

Process multiple P4 XML files and optionally save to CSV.

**Args:**
- `xml_files`: List of Path objects pointing to XML files
- `output_path`: Optional path to save CSV output
- `max_files`: Optional limit for number of files to process

**Returns:** pandas DataFrame

### `extract_metadata(root) -> Dict[str, str]`

Extract bibliographic metadata from P4 XML HEADER element.

**Returns:** Dict with keys: `author`, `place`, `date`

### `extract_pages_by_pb(text_elem, metadata) -> List[Dict]`

Extract page-level text using PB (page break) elements as delimiters.

## License

MIT License - See LICENSE file for details
