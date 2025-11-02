import json
import os
import pathlib
import logging
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Implements sparse retrieval using BM25 on the SciFact dataset.
    Retrieves top 100 documents for each test query and saves results.
    """
    dataset_name = "scifact"
    dataset_path = pathlib.Path("datasets") / dataset_name
    results_dir = pathlib.Path("results")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "sparse_results.json"

    # Load Dataset
    logger.info(f"Loading dataset: {dataset_name}")
    corpus, queries, _ = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
    logger.info("Dataset loaded.")

    # Index Corpus with BM25
    logger.info("Indexing corpus with BM25...")
    
    corpus_ids = list(corpus.keys())
    tokenized_corpus = []
    
    for doc_id in tqdm(corpus_ids, desc="Tokenizing corpus"):
        doc_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
        tokenized_corpus.append(doc_text.lower().split())

    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built.")

    # Retrieve Documents for Queries
    logger.info("Running retrieval for test queries...")
    results = {}
    
    for query_id, query_text in tqdm(queries.items(), desc="Retrieving documents"):
        tokenized_query = query_text.lower().split()
        
        doc_scores = bm25.get_scores(tokenized_query)
        
        scores_with_ids = list(zip(corpus_ids, doc_scores))
        
        scores_with_ids.sort(key=lambda x: x[1], reverse=True)
        
        top_100_scores = scores_with_ids[:100]
        
        results[query_id] = {doc_id: float(score) for doc_id, score in top_100_scores}

    # Save Results
    logger.info(f"Saving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("Sparse retrieval complete.")

if __name__ == "__main__":
    main()
