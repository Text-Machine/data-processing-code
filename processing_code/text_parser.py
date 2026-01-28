"""
Text parser module for converting P4 XML files to structured page-level data.

Supports EEBO (Early English Books Online) and similar TEI P4 XML formats.
Extracts metadata (author, place, date) and page-level text content.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm.auto import tqdm


def extract_metadata(root: ET.Element) -> Dict[str, str]:
    """
    Extract bibliographic metadata from P4 XML HEADER element.
    
    Args:
        root: XML root element (ETS)
    
    Returns:
        Dictionary with keys: author, place, date
    """
    metadata = {
        'author': 'Unknown',
        'place': 'Unknown',
        'date': 'Unknown'
    }
    
    header = root.find('HEADER')
    if header is None:
        return metadata
    
    filedesc = header.find('FILEDESC')
    if filedesc is None:
        return metadata
    
    # Extract author from TITLESTMT
    titlestmt = filedesc.find('TITLESTMT')
    if titlestmt is not None:
        author_elem = titlestmt.find('AUTHOR')
        if author_elem is not None:
            author_text = ' '.join(author_elem.itertext()).strip()
            if author_text:
                metadata['author'] = author_text
    
    # Extract publication info from SOURCEDESC
    sourcedesc = filedesc.find('SOURCEDESC')
    if sourcedesc is not None:
        biblfull = sourcedesc.find('BIBLFULL')
        if biblfull is not None:
            publicationstmt = biblfull.find('PUBLICATIONSTMT')
            if publicationstmt is not None:
                # Extract place
                pubplace = publicationstmt.find('PUBPLACE')
                if pubplace is not None:
                    place_text = ' '.join(pubplace.itertext()).strip()
                    if place_text:
                        metadata['place'] = place_text
                
                # Extract date
                date_elem = publicationstmt.find('DATE')
                if date_elem is not None:
                    date_text = date_elem.text or date_elem.get('VALUE', '')
                    if date_text:
                        metadata['date'] = date_text.strip()
    
    return metadata


def extract_pages_by_pb(text_elem: ET.Element, metadata: Dict[str, str]) -> List[Dict]:
    """
    Extract page-level text using PB (page break) elements as delimiters.
    
    Args:
        text_elem: The TEXT element containing content
        metadata: Dictionary with author, place, date
    
    Returns:
        List of page dictionaries with metadata and page_text
    """
    pages = []
    pb_elements = text_elem.findall('.//PB')
    
    if not pb_elements:
        # No page breaks - treat entire content as one page
        text_content = ' '.join(text_elem.itertext()).strip()
        if text_content:
            pages.append({
                'author': metadata['author'],
                'place': metadata['place'],
                'date': metadata['date'],
                'page_text': text_content
            })
        return pages
    
    # Process text between consecutive page breaks
    for i, pb in enumerate(pb_elements):
        page_text_parts = []
        current = pb
        
        # Collect all text until next PB element
        while True:
            next_elem = None
            
            # Find next sibling
            for p in text_elem.iter():
                children = list(p)
                if current in children:
                    idx = children.index(current)
                    if idx + 1 < len(children):
                        next_elem = children[idx + 1]
                    break
            
            if next_elem is None:
                break
            
            # Stop at next page break
            if next_elem.tag == 'PB':
                break
            
            # Extract text from element
            text = ' '.join(next_elem.itertext()).strip()
            if text:
                page_text_parts.append(text)
            
            current = next_elem
        
        # Create page entry if text exists
        text_content = ' '.join(page_text_parts).strip()
        if text_content:
            pages.append({
                'author': metadata['author'],
                'place': metadata['place'],
                'date': metadata['date'],
                'page_text': text_content
            })
    
    return pages


def parse_xml(xml_path: Path) -> List[Dict]:
    """
    Parse a P4 XML file and extract metadata and page-level text.
    
    P4 XML format structure:
    - ETS (root)
      - HEADER (metadata with author, place, date)
      - EEBO (content)
        - TEXT (contains FRONT, BODY)
          - PB (page break elements)
    
    Args:
        xml_path: Path to P4 XML file
    
    Returns:
        List of dicts with keys: author, place, date, page_text
        Returns empty list if parsing fails
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract metadata
        metadata = extract_metadata(root)
        
        # Extract pages from TEXT element
        pages = []
        eebo = root.find('EEBO')
        if eebo is not None:
            text_elem = eebo.find('TEXT')
            if text_elem is not None:
                pages = extract_pages_by_pb(text_elem, metadata)
        
        return pages
    
    except Exception as e:
        print(f"Error parsing {Path(xml_path).name}: {e}")
        return []


def process_files(
    xml_files: List[Path],
    output_path: Optional[str] = None,
    max_files: Optional[int] = None
) -> pd.DataFrame:
    """
    Process multiple P4 XML files and optionally save to CSV.
    
    Args:
        xml_files: List of Path objects pointing to XML files
        output_path: Optional path to save CSV output
        max_files: Optional limit for number of files to process
    
    Returns:
        pandas DataFrame with columns: author, place, date, page_text
    """
    all_pages = []
    files_to_process = xml_files[:max_files] if max_files else xml_files
    
    print(f"Processing {len(files_to_process)} files...")
    
    for xml_file in tqdm(files_to_process, desc="Parsing XML files"):
        pages = parse_xml(xml_file)
        all_pages.extend(pages)
    
    # Create DataFrame using pandas
    df = pd.DataFrame(all_pages)
    
    print(f"\nTotal pages extracted: {len(df)}")
    
    # Save to CSV if path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
    
    return df
