"""
Sparse Search Engine using Whoosh.

BM25-based keyword search for document retrieval.
"""

import json
import os
from typing import List, Dict, Any, Optional

from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.qparser import QueryParser


class WhooshSearchEngine:
    """Whoosh-based sparse/keyword search engine."""
    
    def __init__(self, ocr_jsonl_path: str, index_dir: Optional[str] = None):
        """
        Initialize sparse search engine.
        
        Args:
            ocr_jsonl_path: Path to OCR JSONL file
            index_dir: Directory to cache index (default: alongside OCR file)
        """
        if index_dir is None:
            index_dir = os.path.join(os.path.dirname(ocr_jsonl_path), "whoosh_index")
        
        self.index_dir = index_dir
        self.ocr_jsonl_path = ocr_jsonl_path
        
        if exists_in(index_dir):
            print(f"Loading cached index from {index_dir}...")
            self.ix = open_dir(index_dir)
            print("Index loaded")
        else:
            self._build_index()
    
    def _build_index(self):
        """Build the search index from OCR JSONL file."""
        os.makedirs(self.index_dir, exist_ok=True)
        
        schema = Schema(
            file=ID(stored=True),
            page_number=NUMERIC(stored=True),
            total_pages=NUMERIC(stored=True),
            text=TEXT(stored=False)
        )
        
        print(f"Building search index from {self.ocr_jsonl_path}...")
        ix = create_in(self.index_dir, schema)
        writer = ix.writer(limitmb=512, procs=1, multisegment=False)
        
        count = 0
        with open(self.ocr_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    page = json.loads(line)
                    writer.add_document(
                        file=page['file'],
                        page_number=page['page_number'],
                        total_pages=page['total_pages'],
                        text=page.get('text', '')
                    )
                    count += 1
        
        writer.commit()
        print(f"Indexed {count} pages (cached to {self.index_dir})")
        self.ix = ix
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching query.
        
        Args:
            query: Keyword search query (supports phrases, AND/OR/NOT, wildcards)
            top_k: Number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        with self.ix.searcher() as searcher:
            query_parser = QueryParser("text", self.ix.schema)
            parsed_query = query_parser.parse(query)
            results = searcher.search(parsed_query, limit=top_k)
            
            return [
                {
                    "file": hit['file'],
                    "page_number": hit['page_number'],
                    "total_pages": hit['total_pages']
                }
                for hit in results
            ]
