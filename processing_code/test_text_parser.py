"""
Unit tests for text_parser module.

Tests cover:
- Metadata extraction from XML headers
- Page-level text extraction
- Handling of edge cases (missing elements, empty files, etc.)
"""

import unittest
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from text_parser import extract_metadata, extract_pages_by_pb, parse_xml, process_files


class TestExtractMetadata(unittest.TestCase):
    """Test metadata extraction from XML headers."""
    
    def setUp(self):
        """Create a basic XML structure for testing."""
        self.root = ET.Element('ETS')
        header = ET.SubElement(self.root, 'HEADER')
        filedesc = ET.SubElement(header, 'FILEDESC')
        
        # Add TITLESTMT with author
        titlestmt = ET.SubElement(filedesc, 'TITLESTMT')
        author = ET.SubElement(titlestmt, 'AUTHOR')
        author.text = 'Test Author'
        
        # Add SOURCEDESC with publication info
        sourcedesc = ET.SubElement(filedesc, 'SOURCEDESC')
        biblfull = ET.SubElement(sourcedesc, 'BIBLFULL')
        publicationstmt = ET.SubElement(biblfull, 'PUBLICATIONSTMT')
        
        pubplace = ET.SubElement(publicationstmt, 'PUBPLACE')
        pubplace.text = 'London'
        
        date = ET.SubElement(publicationstmt, 'DATE')
        date.text = '1649'
    
    def test_extract_metadata_complete(self):
        """Test extraction of complete metadata."""
        metadata = extract_metadata(self.root)
        
        self.assertEqual(metadata['author'], 'Test Author')
        self.assertEqual(metadata['place'], 'London')
        self.assertEqual(metadata['date'], '1649')
    
    def test_extract_metadata_missing_author(self):
        """Test extraction with missing author element."""
        # Remove author
        header = self.root.find('HEADER')
        titlestmt = header.find('.//TITLESTMT')
        author = titlestmt.find('AUTHOR')
        titlestmt.remove(author)
        
        metadata = extract_metadata(self.root)
        
        self.assertEqual(metadata['author'], 'Unknown')
        self.assertEqual(metadata['place'], 'London')
    
    def test_extract_metadata_missing_header(self):
        """Test extraction with missing HEADER element."""
        root = ET.Element('ETS')
        metadata = extract_metadata(root)
        
        self.assertEqual(metadata['author'], 'Unknown')
        self.assertEqual(metadata['place'], 'Unknown')
        self.assertEqual(metadata['date'], 'Unknown')


class TestExtractPagesByPb(unittest.TestCase):
    """Test page extraction using page break elements."""
    
    def setUp(self):
        """Create a TEXT element with page breaks and content."""
        self.metadata = {
            'author': 'Test Author',
            'place': 'London',
            'date': '1649'
        }
    
    def test_extract_pages_with_page_breaks(self):
        """Test extraction of multiple pages separated by PB elements."""
        text_elem = ET.Element('TEXT')
        
        # First page
        p1 = ET.SubElement(text_elem, 'P')
        p1.text = 'First page content'
        
        # Page break
        pb1 = ET.SubElement(text_elem, 'PB')
        pb1.set('N', '2')
        
        # Second page
        p2 = ET.SubElement(text_elem, 'P')
        p2.text = 'Second page content'
        
        pages = extract_pages_by_pb(text_elem, self.metadata)
        
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]['page_text'], 'First page content')
        self.assertEqual(pages[1]['page_text'], 'Second page content')
        self.assertEqual(pages[0]['author'], 'Test Author')
    
    def test_extract_pages_no_page_breaks(self):
        """Test extraction when no PB elements exist."""
        text_elem = ET.Element('TEXT')
        
        p = ET.SubElement(text_elem, 'P')
        p.text = 'All content as one page'
        
        pages = extract_pages_by_pb(text_elem, self.metadata)
        
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]['page_text'], 'All content as one page')
    
    def test_extract_pages_empty_text(self):
        """Test extraction with empty TEXT element."""
        text_elem = ET.Element('TEXT')
        pages = extract_pages_by_pb(text_elem, self.metadata)
        
        self.assertEqual(len(pages), 0)
    
    def test_extract_pages_with_nested_elements(self):
        """Test extraction with nested text elements."""
        text_elem = ET.Element('TEXT')
        
        div = ET.SubElement(text_elem, 'DIV')
        p1 = ET.SubElement(div, 'P')
        p1.text = 'Nested paragraph'
        
        pb = ET.SubElement(text_elem, 'PB')
        
        p2 = ET.SubElement(text_elem, 'P')
        p2.text = 'After page break'
        
        pages = extract_pages_by_pb(text_elem, self.metadata)
        
        self.assertEqual(len(pages), 2)
        self.assertIn('Nested paragraph', pages[0]['page_text'])
        self.assertEqual(pages[1]['page_text'], 'After page break')


class TestParseXml(unittest.TestCase):
    """Test full XML parsing."""
    
    def create_test_xml_file(self, content: str) -> Path:
        """Create a temporary XML file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    def test_parse_complete_document(self):
        """Test parsing a complete P4 XML document."""
        xml_content = '''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT>
                <AUTHOR>William Shakespeare</AUTHOR>
            </TITLESTMT>
            <SOURCEDESC>
                <BIBLFULL>
                    <PUBLICATIONSTMT>
                        <PUBPLACE>London</PUBPLACE>
                        <DATE>1623</DATE>
                    </PUBLICATIONSTMT>
                </BIBLFULL>
            </SOURCEDESC>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>First line of text</P>
            <PB N="2"/>
            <P>Second page text</P>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0]['author'], 'William Shakespeare')
            self.assertEqual(pages[0]['place'], 'London')
            self.assertEqual(pages[0]['date'], '1623')
            self.assertEqual(pages[0]['page_text'], 'First line of text')
            self.assertEqual(pages[1]['page_text'], 'Second page text')
        finally:
            xml_file.unlink()
    
    def test_parse_minimal_document(self):
        """Test parsing a minimal valid P4 XML document."""
        xml_content = '''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT/>
            <SOURCEDESC/>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>Minimal content</P>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0]['author'], 'Unknown')
            self.assertEqual(pages[0]['page_text'], 'Minimal content')
        finally:
            xml_file.unlink()
    
    def test_parse_invalid_file(self):
        """Test parsing of non-existent file."""
        pages = parse_xml(Path('/nonexistent/file.xml'))
        self.assertEqual(pages, [])


class TestProcessFiles(unittest.TestCase):
    """Test batch file processing."""
    
    def create_test_xml_files(self, count: int) -> list:
        """Create multiple test XML files."""
        files = []
        for i in range(count):
            xml_content = f'''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT>
                <AUTHOR>Author {i}</AUTHOR>
            </TITLESTMT>
            <SOURCEDESC>
                <BIBLFULL>
                    <PUBLICATIONSTMT>
                        <PUBPLACE>Place {i}</PUBPLACE>
                        <DATE>160{i}</DATE>
                    </PUBLICATIONSTMT>
                </BIBLFULL>
            </SOURCEDESC>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>Content of document {i}</P>
        </TEXT>
    </EEBO>
</ETS>'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(xml_content)
                files.append(Path(f.name))
        
        return files
    
    def test_process_multiple_files(self):
        """Test processing multiple XML files."""
        xml_files = self.create_test_xml_files(3)
        
        try:
            df = process_files(xml_files)
            
            self.assertEqual(len(df), 3)
            self.assertIn('author', df.columns)
            self.assertIn('place', df.columns)
            self.assertIn('date', df.columns)
            self.assertIn('page_text', df.columns)
        finally:
            for f in xml_files:
                f.unlink()
    
    def test_process_with_max_files(self):
        """Test processing with max_files limit."""
        xml_files = self.create_test_xml_files(5)
        
        try:
            df = process_files(xml_files, max_files=2)
            
            self.assertEqual(len(df), 2)
        finally:
            for f in xml_files:
                f.unlink()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def create_test_xml_file(self, content: str) -> Path:
        """Create a temporary XML file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    def test_parse_with_special_characters(self):
        """Test parsing text with special characters and unicode."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT>
                <AUTHOR>François Rabelais</AUTHOR>
            </TITLESTMT>
            <SOURCEDESC>
                <BIBLFULL>
                    <PUBLICATIONSTMT>
                        <PUBPLACE>Lyon</PUBPLACE>
                        <DATE>1532</DATE>
                    </PUBLICATIONSTMT>
                </BIBLFULL>
            </SOURCEDESC>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>Text with "quotes" and 'apostrophes'</P>
            <PB N="2"/>
            <P>Text with dashes—em-dashes—and ellipsis…</P>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            self.assertEqual(len(pages), 2)
            self.assertIn('François', pages[0]['author'])
            self.assertIn('quotes', pages[0]['page_text'])
            self.assertIn('em-dashes', pages[1]['page_text'])
        finally:
            xml_file.unlink()
    
    def test_parse_with_empty_pages(self):
        """Test handling of empty pages (pages with no text)."""
        xml_content = '''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT/>
            <SOURCEDESC/>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>First page</P>
            <PB N="2"/>
            <PB N="3"/>
            <P>Third page</P>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            # Should skip empty page 2
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0]['page_text'], 'First page')
            self.assertEqual(pages[1]['page_text'], 'Third page')
        finally:
            xml_file.unlink()
    
    def test_parse_with_front_body_structure(self):
        """Test parsing with separate FRONT and BODY sections."""
        xml_content = '''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT>
                <AUTHOR>John Donne</AUTHOR>
            </TITLESTMT>
            <SOURCEDESC>
                <BIBLFULL>
                    <PUBLICATIONSTMT>
                        <PUBPLACE>London</PUBPLACE>
                        <DATE>1633</DATE>
                    </PUBLICATIONSTMT>
                </BIBLFULL>
            </SOURCEDESC>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <FRONT>
                <P>Preface page 1</P>
                <PB N="2"/>
                <P>Preface page 2</P>
            </FRONT>
            <BODY>
                <PB N="3"/>
                <P>Main text begins</P>
            </BODY>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            self.assertEqual(len(pages), 3)
            self.assertIn('Preface', pages[0]['page_text'])
            self.assertIn('Main text', pages[2]['page_text'])
        finally:
            xml_file.unlink()
    
    def test_parse_with_nested_formatting(self):
        """Test parsing with nested formatting elements."""
        xml_content = '''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT/>
            <SOURCEDESC/>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>Text with <I>italic</I> and <B>bold</B> formatting</P>
            <PB N="2"/>
            <DIV>
                <HEAD>Chapter 1</HEAD>
                <P>Chapter content with <QUOTE>quoted text</QUOTE></P>
            </DIV>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            self.assertEqual(len(pages), 2)
            self.assertIn('italic', pages[0]['page_text'])
            self.assertIn('bold', pages[0]['page_text'])
            self.assertIn('Chapter', pages[1]['page_text'])
            self.assertIn('quoted', pages[1]['page_text'])
        finally:
            xml_file.unlink()
    
    def test_parse_with_whitespace_preservation(self):
        """Test that whitespace is properly handled."""
        xml_content = '''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT/>
            <SOURCEDESC/>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>Line  one  with  extra  spaces</P>
            <P>
                Line two with
                newlines and
                indentation
            </P>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            # Should preserve text but normalize whitespace through join
            self.assertEqual(len(pages), 1)
            text = pages[0]['page_text']
            # Text should be combined and properly spaced
            self.assertIn('Line', text)
        finally:
            xml_file.unlink()
    
    def test_metadata_with_multiple_authors(self):
        """Test metadata extraction with multiple authors."""
        root = ET.Element('ETS')
        header = ET.SubElement(root, 'HEADER')
        filedesc = ET.SubElement(header, 'FILEDESC')
        
        titlestmt = ET.SubElement(filedesc, 'TITLESTMT')
        author1 = ET.SubElement(titlestmt, 'AUTHOR')
        author1.text = 'Author One'
        author2 = ET.SubElement(titlestmt, 'AUTHOR')
        author2.text = 'Author Two'
        
        metadata = extract_metadata(root)
        
        # Should capture first author
        self.assertEqual(metadata['author'], 'Author One')
    
    def test_parse_with_date_attribute(self):
        """Test parsing when date is in attribute instead of text."""
        root = ET.Element('ETS')
        header = ET.SubElement(root, 'HEADER')
        filedesc = ET.SubElement(header, 'FILEDESC')
        
        titlestmt = ET.SubElement(filedesc, 'TITLESTMT')
        author = ET.SubElement(titlestmt, 'AUTHOR')
        author.text = 'Test Author'
        
        sourcedesc = ET.SubElement(filedesc, 'SOURCEDESC')
        biblfull = ET.SubElement(sourcedesc, 'BIBLFULL')
        publicationstmt = ET.SubElement(biblfull, 'PUBLICATIONSTMT')
        
        date_elem = ET.SubElement(publicationstmt, 'DATE')
        date_elem.set('VALUE', '1650')
        
        metadata = extract_metadata(root)
        
        self.assertEqual(metadata['date'], '1650')


class TestIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios."""
    
    def create_test_xml_file(self, content: str) -> Path:
        """Create a temporary XML file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    def test_realistic_eebo_document(self):
        """Test parsing a realistic EEBO document structure."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT>
                <TITLE>A Tale of Two Cities</TITLE>
                <AUTHOR>Charles Dickens</AUTHOR>
            </TITLESTMT>
            <SOURCEDESC>
                <BIBLFULL>
                    <TITLEPAGE>
                        <P>A Tale of Two Cities</P>
                    </TITLEPAGE>
                    <PUBLICATIONSTMT>
                        <PUBPLACE>London</PUBPLACE>
                        <DATE>1859</DATE>
                    </PUBLICATIONSTMT>
                </BIBLFULL>
            </SOURCEDESC>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <IDG S="marc" R="UM" ID="000000001">
            <STC>STC123</STC>
            <BIBNO>12345678</BIBNO>
        </IDG>
        <TEXT LANG="eng">
            <FRONT>
                <HEAD>Preface</HEAD>
                <P>It was the best of times, it was the worst of times</P>
                <PB N="2" REF="image002.tif"/>
                <P>by Charles Dickens</P>
            </FRONT>
            <BODY>
                <PB N="3" REF="image003.tif"/>
                <BOOK N="1">
                    <CHAPTER>
                        <HEAD>Chapter 1: The Period</HEAD>
                        <P>It was the year of Our Lord one thousand seven hundred and seventy-five.</P>
                    </CHAPTER>
                </BOOK>
            </BODY>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_file = self.create_test_xml_file(xml_content)
        
        try:
            pages = parse_xml(xml_file)
            
            # Should extract 3 pages
            self.assertGreater(len(pages), 0)
            
            # Check metadata
            first_page = pages[0]
            self.assertEqual(first_page['author'], 'Charles Dickens')
            self.assertEqual(first_page['place'], 'London')
            self.assertEqual(first_page['date'], '1859')
            
            # Check content
            combined_text = ' '.join([p['page_text'] for p in pages])
            self.assertIn('best of times', combined_text)
            self.assertIn('Chapter 1', combined_text)
        finally:
            xml_file.unlink()
    
    def test_batch_processing_consistency(self):
        """Test that batch processing produces consistent results."""
        xml_content = '''<?xml version="1.0"?>
<ETS>
    <HEADER>
        <FILEDESC>
            <TITLESTMT>
                <AUTHOR>Test Author</AUTHOR>
            </TITLESTMT>
            <SOURCEDESC>
                <BIBLFULL>
                    <PUBLICATIONSTMT>
                        <PUBPLACE>Test Place</PUBPLACE>
                        <DATE>1700</DATE>
                    </PUBLICATIONSTMT>
                </BIBLFULL>
            </SOURCEDESC>
        </FILEDESC>
    </HEADER>
    <EEBO>
        <TEXT>
            <P>Test content</P>
        </TEXT>
    </EEBO>
</ETS>'''
        
        xml_files = []
        for i in range(3):
            xml_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.xml', 
                delete=False
            )
            xml_file.write(xml_content)
            xml_file.close()
            xml_files.append(Path(xml_file.name))
        
        try:
            df = process_files(xml_files)
            
            # Should have 3 rows (one page per file)
            self.assertEqual(len(df), 3)
            
            # All rows should have same author
            self.assertTrue((df['author'] == 'Test Author').all())
            
            # All should have same metadata
            self.assertTrue((df['place'] == 'Test Place').all())
            self.assertTrue((df['date'] == '1700').all())
        finally:
            for f in xml_files:
                f.unlink()


if __name__ == '__main__':
    unittest.main()
