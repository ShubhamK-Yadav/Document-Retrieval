# Document Retrieval System - CACM Dataset
This project implements a traditional Information Retrieval (IR) system using the **Vector Space Model**.
It processes a collection of CACM (Communications of the ACM) documents, builds an inverted index, and retrieves the most relevant documents for a given query.
The system supports **multiple term weighting schemes**, **configurable preprocessing**, and **pseudo-relevance feedback (PRF)** for improved search performance.

### Key Features 
- Built an inverted index over the CACM collection with configurable preprocessing options (stopword removal, stemming).
- Implemented retrieval algorithms supporting binary, TF, and TF–IDF term weighting schemes.
- Added pseudo-relevance feedback to improve query results.
- Developed the `Retrieve` class to rank documents based on relevance, replacing the provided placeholder logic.
- Evaluated system performance against a gold standard using standard IR metrics via a benchmarking script.

### Installation
Clone the repository and navigate into the project folder:
```
git clone https://github.com/your-username/document-retrieval-system.git
cd document-retrieval-system
```

### Usage
Run `IR_engine.py` with desired options:
```
python IR_engine.py -w tfidf -s -t -o results.txt
```

**Options:**
- `-w` – Term weighting scheme: binary, tf, tfidf
- `-s` – Apply stopword removal
- `-t` – Apply stemming
- `-o` – Output results file name
- `-f` – Enable pseudo-relevance feedback (PRF)

**Example with PRF enabled:**
```
python IR_engine.py -w tfidf -s -t -f -o prf_results.txt
```

**Example of Evaluating performance:**
Compare your results to the gold standard:
```
python eval_ir.py cacm_gold_std.txt results.txt
```
