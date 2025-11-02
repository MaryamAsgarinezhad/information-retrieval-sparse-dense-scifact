import json
import os
import pathlib
import logging
import numpy as np
import faiss
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Implements dense retrieval using Sentence-Transformers and FAISS on SciFact.
    Retrieves top 100 documents for each test query and saves results.
    """
    dataset_name = "scifact"
    dataset_path = pathlib.Path("datasets") / dataset_name
    results_dir = pathlib.Path("results")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "dense_results.json"
    
    model_name = 'all-MiniLM-L6-v2'

    # Load Dataset
    logger.info(f"Loading dataset: {dataset_name}")
    corpus, queries, _ = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
    logger.info("Dataset loaded.")

    # Load SentenceTransformer Model
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded.")

    # Embed Corpus and Build FAISS Index
    logger.info("Embedding corpus...")
    
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "") 
        for doc_id in corpus_ids
    ]

    corpus_embeddings = model.encode(
        corpus_texts, 
        convert_to_tensor=False, 
        show_progress_bar=True
    )
    
    faiss.normalize_L2(corpus_embeddings)
    
    logger.info("Corpus embedded. Building FAISS index...")
    
    dimension = corpus_embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    
    index.add(corpus_embeddings)
    
    logger.info(f"FAISS index built. Total documents indexed: {index.ntotal}")

    # Retrieve Documents for Queries
    logger.info("Running retrieval for test queries...")
    results = {}
    
    for query_id, query_text in tqdm(queries.items(), desc="Retrieving documents"):
        query_embedding = model.encode(query_text, convert_to_tensor=False)
        
        query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding, 100)
        
        top_k_scores = scores[0]
        top_k_indices = indices[0]
        
        doc_ids = [corpus_ids[i] for i in top_k_indices]
        
        results[query_id] = {
            doc_id: float(score) 
            for doc_id, score in zip(doc_ids, top_k_scores)
        }

    # Save Results
    logger.info(f"Saving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("Dense retrieval complete.")

if __name__ == "__main__":
    main()
