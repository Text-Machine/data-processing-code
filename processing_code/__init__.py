"""
Processing Code Package

A Python package for converting P4 XML files (EEBO, ECCO, EVAN) to structured 
page-level CSV data for text mining and machine learning applications.

Main exports:
- parse_xml: Parse a single P4 XML file
- process_files: Process multiple P4 XML files to DataFrame
- extract_metadata: Extract bibliographic metadata from XML
- extract_pages_by_pb: Extract page-level text using page break delimiters
"""

from .text_parser import (
    parse_xml,
    process_files,
    extract_metadata,
    extract_pages_by_pb,
)

__version__ = "0.1.0"
__author__ = "Text-Machine"

__all__ = [
    'parse_xml',
    'process_files',
    'extract_metadata',
    'extract_pages_by_pb',
]
