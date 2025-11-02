# R2L Lab - Part 2: Implementation Challenge

This project implements and evaluates two fundamental information retrieval systems, Sparse (BM25) and Dense (Sentence-Transformers + FAISS), on the SciFact dataset from the BEIR benchmark.
## Project Structure

```

.

├── datasets/                 # (Created by download_data.py)

│   └── scifact/

├── results/                  # (Created by retrieval scripts)

│   ├── dense_results.json

│   └── sparse_results.json

├── download_data.py          # Script to download the SciFact dataset

├── sparse_retrieval.py       # Script to run BM25 retrieval

├── dense_retrieval.py        # Script to run Dense retrieval (S-BERT + FAISS)

├── evaluation.py             # The provided evaluation script

└── requirements.txt          # Python dependencies

```


## Setup & Run Instructions

Follow these steps to set up the environment, run the retrieval pipelines, and evaluate the results.

### 1. Set Up Environment

First, create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data

Run the download script. This will download the SciFact dataset and unzip it into the `datasets/` directory.

```bash
python download_data.py
```
This will create `datasets/scifact`.

### 3. Run Retrieval Pipelines

Run the scripts for both sparse and dense retrieval. These scripts will load the corpus, run retrieval for all test queries, and save the formatted results to the `results/` directory.

```bash
# Run Sparse Retrieval (BM25)
# This will create results/sparse_results.json
python sparse_retrieval.py

# Run Dense Retrieval (Sentence-Transformer + FAISS)
# This will create results/dense_results.json
python dense_retrieval.py
```
### 4. Evaluate the Retrievers

Use the provided `evaluation.py` script to evaluate the generated results files against the SciFact QRELS (ground truth).

```bash
# Evaluate the sparse retriever
python evaluation.py datasets/scifact results/sparse_results.json

# Evaluate the dense retriever
python evaluation.py datasets/scifact results/dense_results.json
```
## Results and Discussion

### Expected Output

Running the evaluation will produce output similar to this:

```
--- Evaluation for results/sparse_results.json ---

nDCG@10: 0.5597
Recall@100: 0.7929
MRR: 0.5290

--- Evaluation for results/dense_results.json ---

nDCG@10: 0.6451
Recall@100: 0.9250
MRR: 0.6110
```

### Discussion

**Which retriever performed better?**

The Dense Retriever (all-MiniLM-L6-v2 + FAISS) performed better than the Sparse Retriever (BM25) across all key metrics (nDCG@10, Recall@100, and MRR).

The SciFact dataset involves matching a short scientific "claim" (the query) to a scientific abstract (the document) that either supports or refutes it. This task is highly dependent on semantic understanding, not just keyword overlap.

* **Semantic Mismatch:** A claim like "X is caused by Y" might be supported by a document that says "Y is a primary factor in the etiology of X." These two sentences share few keywords, but their meaning is identical.

  BM25 (Sparse) would likely fail here as it relies on term-matching (like "caused" and "etiology"). Dense Retrieval (S-BERT) excels at this. It's trained to understand that these different phrases have similar meanings and maps them to nearby vectors in the embedding space.

* **Domain-Specific Terminology:** While BM25 is good with specific terms, the S-BERT model has been pre-trained on a massive corpus and fine-tuned on question-answering pairs, giving it a robust understanding of how terms relate, even complex scientific ones.


### Performance Trade-offs

|Aspect|Sparse (BM25)|Dense (S-BERT + FAISS)|
|---|---|---|
|Retrieval Quality|Lower. Good at keyword matching but fails on semantic mismatch.|Higher. Excellent at semantic matching, synonyms, and paraphrasing.|
|Indexing Speed|Extremely Fast. Indexing is just fast text tokenization.|Very Slow. Must pass the entire corpus through a neural network (embedding).|
|Indexing Cost|Low. Requires CPU and moderate RAM.|High. Requires a powerful CPU (or GPU) and significant time.|
|Index Size / Memory|Low. The index is relatively small.|High. Must store a high-dimensional vector for every document.|
|Retrieval Speed|Very Fast. Inverted index lookups are highly optimized.|Fast (but slower than BM25). ANN search with FAISS is fast, but still involves vector math.|
|Interpretability|High. You can see which keywords contributed to the score.|Low. It's difficult to know why two vectors are "close."|
