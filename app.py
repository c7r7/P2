"""
@author: Sougata Saha
@edited_by: Charan Kumar Raju
Institute: University at Buffalo
Description:
Flask server for Project 2 (CSE 4/535) — Boolean Query Processing and Inverted Index
"""

from flask import Flask, request, jsonify
from P2.preprocess import Preprocessor
from indexer import Indexer
from daat import daat_and, daat_and_with_skips, rank_by_tfidf

app = Flask(__name__)

# -------------------------
# Build the inverted index
# -------------------------
print("Building inverted index...")

pre = Preprocessor()
indexer = Indexer()

@app.route('/')
def home():
    return """
    ✅ Flask app is running!<br>
    Use POST /execute_query with a JSON body.<br><br>
    Example using curl:<br>
    <code>
    curl -X POST http://127.0.0.1:9999/execute_query \
      -H "Content-Type: application/json" \
      -d '{"queries": ["information retrieval", "boolean model", "text mining"]}'
    </code>
    """

with open("data/input_corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        doc_id, text = pre.get_doc_id(line)
        tokens = pre.tokenizer(text)
        indexer.generate_inverted_index(doc_id, tokens)

indexer.sort_terms()
indexer.add_skip_connections()
indexer.calculate_tf_idf()

print("✅ Index built successfully!")

# -------------------------
# Define the API endpoint
# -------------------------
@app.route('/execute_query', methods=['POST'])
def execute_query():
    """
    Input JSON:
    {
        "queries": [
            "information retrieval",
            "boolean model inverted index",
            "text preprocessing steps"
        ]
    }

    Output JSON:
    Matches structure in sample_output.json
    """
    data = request.get_json()
    queries = data.get("queries", [])

    inverted_index = indexer.get_index()
    results = {}

    for query in queries:
        query_terms = pre.tokenizer(query)
        term_postings = {}

        # Step 1: collect postings list for each term
        for term in query_terms:
            postings = []
            skips = []
            if term in inverted_index:
                plist = inverted_index[term]
                postings = plist.get_all_doc_ids()
                skips = plist.get_skip_doc_ids()
            term_postings[term] = {
                "postingsList": postings,
                "postingsListSkip": skips
            }

        # Step 2: DAAT (without and with skips)
        daat_result, daat_comparisons = daat_and(query_terms, inverted_index)
        daat_skip_result, daat_skip_comparisons = daat_and_with_skips(query_terms, inverted_index)

        # Step 3: TF-IDF ranking of results
        tfidf_ranked = rank_by_tfidf(daat_result, query_terms, inverted_index)
        tfidf_skip_ranked = rank_by_tfidf(daat_skip_result, query_terms, inverted_index)

        # Step 4: Combine into output JSON structure
        results[query] = {
            "postingsList": {term: term_postings[term]["postingsList"] for term in query_terms},
            "postingsListSkip": {term: term_postings[term]["postingsListSkip"] for term in query_terms},
            "daatAnd": {
                "results": daat_result,
                "num_docs": len(daat_result),
                "num_comparisons": daat_comparisons
            },
            "daatAndSkip": {
                "results": daat_skip_result,
                "num_docs": len(daat_skip_result),
                "num_comparisons": daat_skip_comparisons
            },
            "daatAndTfIdf": tfidf_ranked,
            "daatAndSkipTfIdf": tfidf_skip_ranked
        }

    return jsonify(results)


# -------------------------
# Run the Flask app
# -------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
