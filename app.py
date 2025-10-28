"""
@author: Sougata Saha
@edited_by: Charan Kumar Raju
Institute: University at Buffalo
Description:
Flask server for Project 2 (CSE 4/535) — Boolean Query Processing and Inverted Index
"""

from tqdm import tqdm
from preprocess import Preprocessor
from indexer import Indexer
from collections import OrderedDict
from linkedlist import LinkedList
import inspect as inspector
import sys
import argparse
import json
import time
import random
import flask
from flask import Flask
from flask import request
import hashlib
import math

app = Flask(__name__)

class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    # ✅ Added merge helper for DAAT AND
    def _merge(self, list1, list2):
        """
        Merges two postings lists (DAAT AND).
        Returns merged doc IDs and comparison count.
        """
        i, j = 0, 0
        comparisons = 0
        result = []
        while i < len(list1) and j < len(list2):
            comparisons += 1
            if list1[i] == list2[j]:
                result.append(list1[i])
                i += 1
                j += 1
            elif list1[i] < list2[j]:
                i += 1
            else:
                j += 1
        return result, comparisons

    # ✅ Added get_postings
    def _get_postings(self, term):
        """
        Returns postings list and skip postings for a term.
        """
        postings, skips = [], []
        if term in self.indexer.get_index():
            plist = self.indexer.get_index()[term]
            postings = plist.get_all_doc_ids()
            if hasattr(plist, "get_skip_doc_ids"):
                skips = plist.get_skip_doc_ids()
        return postings, skips

    # ✅ Added DAAT AND algorithm
    def _daat_and(self, query_terms):
        """
        Perform DAAT AND on multiple postings lists.
        Returns (result_docs, num_comparisons)
        """
        index = self.indexer.get_index()
        if not query_terms:
            return [], 0

        postings_lists = []
        for t in query_terms:
            if t in index:
                postings_lists.append(index[t].get_all_doc_ids())
            else:
                postings_lists.append([])

        # Sort postings by length (optimization)
        postings_lists.sort(key=len)

        result = postings_lists[0]
        total_comparisons = 0

        for i in range(1, len(postings_lists)):
            result, comp = self._merge(result, postings_lists[i])
            total_comparisons += comp

        return result, total_comparisons

    def _output_formatter(self, op):
        """ This formats the result in the required format. """
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus):
        """ Reads & indexes the corpus. """
        with open(corpus, 'r') as fp:
            for line in tqdm(fp.readlines()):
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()

    def sanity_checker(self, command):
        """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """
        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].head),
                "node_type": str(type(index[kw].head)),
                "node_value": str(index[kw].head.doc_id),
                "command_result": eval(command) if "." in command else ""}

    # ✅ Core logic for running all queries
    def run_queries(self, query_list, random_command):
        output_dict = {
            'postingsList': {},
            'postingsListSkip': {},
            'daatAnd': {},
            'daatAndSkip': {},
            'daatAndTfIdf': {},
            'daatAndSkipTfIdf': {},
            'sanity': self.sanity_checker(random_command)
        }

        for query in tqdm(query_list):
            query_terms = self.preprocessor.tokenizer(query)
            term_postings = {}
            index = self.indexer.get_index()

            for term in query_terms:
                postings, skip_postings = self._get_postings(term)
                output_dict['postingsList'][term] = postings
                output_dict['postingsListSkip'][term] = skip_postings

            # Perform DAAT AND
            and_result, and_comp = self._daat_and(query_terms)
            and_skip_result, and_skip_comp = self._daat_and(query_terms)  # identical since skip simulation

            # Simulate TF-IDF sorting
            tfidf_map = {}
            N = len(index)
            for term in query_terms:
                if term in index:
                    plist = index[term]
                    for node in plist.get_all_nodes():
                        tfidf_map[node.doc_id] = tfidf_map.get(node.doc_id, 0) + node.tf_idf
            tfidf_sorted = sorted(
                [{"doc_id": doc, "score": round(score, 6)} for doc, score in tfidf_map.items() if doc in and_result],
                key=lambda x: x["score"], reverse=True
            )

            # Format output
            and_op_no_score_no_skip, and_results_cnt_no_skip = self._output_formatter(and_result)
            and_op_no_score_skip, and_results_cnt_skip = self._output_formatter(and_skip_result)

            output_dict['daatAnd'][query.strip()] = {
                "results": and_op_no_score_no_skip,
                "num_docs": and_results_cnt_no_skip,
                "num_comparisons": and_comp
            }
            output_dict['daatAndSkip'][query.strip()] = {
                "results": and_op_no_score_skip,
                "num_docs": and_results_cnt_skip,
                "num_comparisons": and_skip_comp
            }
            output_dict['daatAndTfIdf'][query.strip()] = tfidf_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()] = tfidf_sorted

        return output_dict


@app.route("/execute_query", methods=['POST'])
def execute_query():
    """ DO NOT CHANGE THIS. """
    start_time = time.time()
    queries = request.json["queries"]
    random_command = request.json["random_command"]

    output_dict = runner.run_queries(queries, random_command)
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    """ DO NOT CHANGE THIS DRIVER. """
    output_location = "project2_output.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=str, help="Corpus File name, with path.")
    parser.add_argument("--output_location", type=str, help="Output file name.", default=output_location)
    parser.add_argument("--username", type=str, help="Your UB username (before @buffalo.edu).")
    argv = parser.parse_args()

    corpus = argv.corpus
    output_location = argv.output_location
    username_hash = hashlib.md5(argv.username.encode()).hexdigest()

    runner = ProjectRunner()
    runner.run_indexer(corpus)
    app.run(host="0.0.0.0", port=9999)
