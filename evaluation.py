import sys
import json
import pathlib
import logging
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Evaluates retrieval results and prints:
        • nDCG@10
        • Recall@100
        • MRR   (full MRR, no k-cutoff)
    """
    if len(sys.argv) != 3:
        print("Usage: python evaluation.py <path_to_dataset_dir> <path_to_results_file>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    results_file = sys.argv[2]

    if not pathlib.Path(dataset_path).exists():
        logger.error(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)
    if not pathlib.Path(results_file).exists():
        logger.error(f"Error: Results file not found: {results_file}")
        sys.exit(1)


    # Load dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    _, _, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")

    # Load retrieval results
    logger.info(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Standard BEIR metrics (NDCG, Recall, …)
    logger.info("Running standard BEIR evaluation …")
    k_values = [1, 3, 5, 10, 100]
    evaluator = EvaluateRetrieval()

    ndcg, _map, recall, _ = evaluator.evaluate(
        qrels=qrels, results=results, k_values=k_values
    )

    logger.info("Computing full MRR (no k-cutoff) …")
    mrr_full = 0.0
    n_queries = len(qrels)

    for qid, rel in qrels.items():
        ranked = sorted(results.get(qid, {}).items(),
                        key=lambda x: x[1], reverse=True)

        first_rel_rank = None
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            if doc_id in rel and rel[doc_id] > 0:
                first_rel_rank = rank
                break

        recip = 1.0 / first_rel_rank if first_rel_rank is not None else 0.0
        mrr_full += recip

    mrr_full /= n_queries

    # Print results 
    ndcg_10    = ndcg.get("NDCG@10", 0.0)
    recall_100 = recall.get("Recall@100", 0.0)

    print(f"\n--- Evaluation for {pathlib.Path(results_file).name} ---")
    print(f"nDCG@10: {ndcg_10:.4f}")
    print(f"Recall@100: {recall_100:.4f}")
    print(f"MRR: {mrr_full:.4f}")
    print("----------------------------------------\n")

if __name__ == "__main__":
    main()