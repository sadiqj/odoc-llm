#!/usr/bin/env python3
"""
Unified search for OCaml modules using both BM25 and embedding search.

This script combines sparse (BM25) and dense (embedding) retrieval methods
to provide comprehensive search results across OCaml module documentation.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import bm25s
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class UnifiedSearchEngine:
    def __init__(self, embedding_dir: str = "package-embeddings", 
                 index_dir: str = "module-indexes",
                 model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
        self.embedding_dir = Path(embedding_dir)
        self.index_dir = Path(index_dir)
        self.embeddings = {}
        self.metadata = {}
        self.bm25_indexes = {}
        self.module_paths = {}
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
            
        self.max_length = 8192
        
    def load_package_data(self, package_names: List[str]) -> None:
        """Load embeddings and BM25 indexes for specified packages."""
        print(f"Loading data for {len(package_names)} packages...")
        
        for package in tqdm(package_names, desc="Loading packages"):
            # Load embeddings
            embedding_path = self.embedding_dir / "packages" / package / "embeddings.npz"
            metadata_path = self.embedding_dir / "packages" / package / "metadata.json"
            
            if embedding_path.exists() and metadata_path.exists():
                # Load embeddings
                data = np.load(embedding_path)
                package_embeddings = data['embeddings']
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    package_metadata = json.load(f)
                
                # Store with package prefix in module paths
                for i, module_info in enumerate(package_metadata['modules']):
                    full_path = f"{package}::{module_info['module_path']}"
                    self.embeddings[full_path] = package_embeddings[i]
                    self.metadata[full_path] = {
                        'package': package,
                        'module_path': module_info['module_path'],
                        'description': module_info.get('description', '')
                    }
            
            # Load BM25 index
            index_path = self.index_dir / package
            if index_path.exists():
                # Load the BM25 index
                retriever = bm25s.BM25.load(str(index_path / "index"))
                self.bm25_indexes[package] = retriever
                
                # Load module paths for this package
                with open(index_path / "module_paths.json", 'r') as f:
                    self.module_paths[package] = json.load(f)
    
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last token from the hidden states."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query string using local model."""
        # Tokenize the query
        batch_dict = self.tokenizer([query], max_length=self.max_length, 
                                   padding=True, truncation=True, return_tensors='pt')
        
        # Move to GPU if available
        if torch.cuda.is_available():
            batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings[0].cpu().numpy()
    
    def embedding_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Perform embedding-based semantic search."""
        if not self.embeddings:
            return []
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        results = []
        for module_path, module_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, module_embedding)
            results.append((module_path, similarity, self.metadata[module_path]))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def bm25_search(self, query: str, packages: List[str], top_k: int = 10) -> List[Tuple[str, float, str]]:
        """Perform BM25 search across specified packages."""
        all_results = []
        
        for package in packages:
            if package not in self.bm25_indexes:
                continue
            
            retriever = self.bm25_indexes[package]
            module_paths = self.module_paths[package]
            
            # Tokenize query
            query_tokens = bm25s.tokenize([query], stopwords="en")
            
            # Search
            results, scores = retriever.retrieve(query_tokens, k=min(top_k, len(module_paths)))
            
            # Convert results to module paths with scores
            for i, (doc_ids, doc_scores) in enumerate(zip(results, scores)):
                for doc_id, score in zip(doc_ids, doc_scores):
                    if 0 <= doc_id < len(module_paths):
                        full_path = f"{package}::{module_paths[doc_id]}"
                        all_results.append((full_path, float(score), package))
        
        # Sort all results by score and return top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    def unified_search(self, query: str, packages: List[str], top_k: int = 10) -> Dict:
        """Perform unified search combining BM25 and embedding results."""
        # Load data for specified packages if not already loaded
        packages_to_load = [p for p in packages if p not in self.bm25_indexes]
        if packages_to_load:
            self.load_package_data(packages_to_load)
        
        # Perform both searches
        embedding_results = self.embedding_search(query, top_k)
        bm25_results = self.bm25_search(query, packages, top_k)
        
        # Filter embedding results to only include specified packages
        embedding_results = [
            (path, score, meta) for path, score, meta in embedding_results
            if meta['package'] in packages
        ][:top_k]
        
        # Deduplicate results
        seen_modules = set()
        deduplicated_embedding = []
        deduplicated_bm25 = []
        
        for path, score, meta in embedding_results:
            if path not in seen_modules:
                seen_modules.add(path)
                deduplicated_embedding.append({
                    'module_path': path,
                    'score': float(score),
                    'package': meta['package'],
                    'module_name': meta['module_path'],
                    'description': meta['description']
                })
        
        for path, score, package in bm25_results:
            if path not in seen_modules:
                seen_modules.add(path)
                deduplicated_bm25.append({
                    'module_path': path,
                    'score': score,
                    'package': package,
                    'module_name': path.split('::')[1] if '::' in path else path
                })
        
        return {
            'query': query,
            'packages_searched': packages,
            'embedding_results': deduplicated_embedding,
            'bm25_results': deduplicated_bm25
        }


def main():
    parser = argparse.ArgumentParser(description="Unified search for OCaml modules")
    parser.add_argument("query", help="Search query")
    parser.add_argument("packages", nargs="+", help="Packages to search in")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--embedding-dir", default="package-embeddings",
                        help="Directory containing package embeddings")
    parser.add_argument("--index-dir", default="module-indexes",
                        help="Directory containing BM25 indexes")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format")
    
    args = parser.parse_args()
    
    # Create search engine
    engine = UnifiedSearchEngine(
        embedding_dir=args.embedding_dir,
        index_dir=args.index_dir
    )
    
    # Perform search
    results = engine.unified_search(args.query, args.packages, args.top_k)
    
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(f"\n🔍 Search results for: '{args.query}'")
        print(f"📦 Packages searched: {', '.join(args.packages)}")
        
        print(f"\n🧠 Semantic Search Results (Embeddings):")
        print("─" * 80)
        for i, result in enumerate(results['embedding_results'], 1):
            print(f"{i}. {result['package']}::{result['module_name']} (score: {result['score']:.4f})")
            if result['description']:
                print(f"   {result['description'][:150]}...")
            print()
        
        print(f"\n📝 Full-Text Search Results (BM25):")
        print("─" * 80)
        for i, result in enumerate(results['bm25_results'], 1):
            print(f"{i}. {result['package']}::{result['module_name']} (score: {result['score']:.4f})")
            print()


if __name__ == "__main__":
    main()