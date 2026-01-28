# Text-Machine Project: AI Agent Instructions

## Project Overview
This is a Python-based data processing project focused on working with EEBO (Early English Books Online) text data. The project uses Jupyter notebooks for exploratory data analysis and processing workflows.

## Project Structure
```
/
├── eebo.ipynb              # Main notebook for EEBO data processing
└── data-processing-code/   # Core data processing utilities and scripts
    ├── README.md
    ├── LICENSE (MIT)
    └── .gitignore
```

## Development Environment

### Python Environment
- **Primary environment**: Conda environment named `tm`
- Activate before working: `conda activate tm`
- The project uses Jupyter notebooks for data processing workflows

### Key Technologies
- Python (Jupyter notebooks for analysis and processing)
- **pandas** for data manipulation and CSV operations (preferred for all data processing)
- Conda for environment management
- Git for version control

## Development Workflow

### Working with Notebooks
- Main notebook: [eebo.ipynb](eebo.ipynb) for EEBO data processing
- Currently empty/early stage - expect to build data loading, cleaning, and analysis pipelines
- Use Jupyter notebook interface or VS Code notebook editor

### Environment Setup
```bash
# Always activate the tm conda environment first
conda activate tm

# Navigate to project root or data-processing-code as needed
cd /Users/kasparbeelen/Documents/Text-Machine
```

## Project Conventions

### Code Organization
- Notebooks at root level for exploratory analysis and processing scripts
- `data-processing-code/` contains experimental code and prototypes
- Follow standard Python conventions per `.gitignore` (PEP8, standard project structure)
- **Prefer pandas** for all data manipulation, reading/writing CSV files, and DataFrame operations

### Version Control
- Standard Python `.gitignore` includes common patterns:
  - `__pycache__/`, `*.pyc`, `.ipynb_checkpoints`
  - Virtual environments (`.venv`, `venv/`, `.env`)
  - IDE configs (`.vscode/`, `.idea/`, `.cursor*`)
  - Build artifacts and distribution files
- Nested git structure: `data-processing-code/` has its own git repository

## Working with EEBO Data

### Source Data
- **Location**: `/Volumes/WorkData/Text_Machine_Data/eebo_all/eebo_phase1/P4_XML_TCP`
- **Format**: P4 XML files (TEI P4 encoding standard)
- EEBO Phase 1 collection from TCP (Text Creation Partnership)

### Data Processing Pipeline
- **Goal**: Extract EEBO text content with minimal metadata for downstream analysis
- **Output format**: CSV file with page-level granularity
- **Required columns**:
  - `author`: Book author(s)
  - `place`: Place of publication
  - `date`: Date of publication
  - Page content (text from each page)
- Each row represents **one page** from EEBO texts

### Expected Output
- **Primary deliverable**: CSV file optimized for text mining and machine learning
- Used for downstream ML applications and computational text analysis
- Page-level granularity enables fine-grained historical text analysis

## License
MIT License - Copyright (c) 2026 Text-Machine. See [LICENSE](data-processing-code/LICENSE) for details.
