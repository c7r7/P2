'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from linkedlist import LinkedList
from collections import OrderedDict
import math


class Indexer:
    def __init__(self):
        """Initialize the inverted index and any helper structures."""
        self.inverted_index = OrderedDict({})
        # optional: store document lengths (token counts) for tf normalization
        self.doc_token_counts = {}

    def get_index(self):
        """Return the inverted index (already implemented)."""
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """
        Add each tokenized document to the index.
        Also track document length for TF normalization.
        """
        # store total tokens in doc for later TF computation
        self.doc_token_counts[doc_id] = len(tokenized_document)
        for t in tokenized_document:
            self.add_to_index(t, doc_id)

    def add_to_index(self, term_, doc_id_):
        """
        Adds the given term and its doc_id to the inverted index.
        - If term not present, create a new LinkedList and insert doc_id.
        - If term already present, insert doc_id in sorted order if not duplicate.
        """
        if term_ not in self.inverted_index:
            ll = LinkedList()
            ll.insert(doc_id_)
            self.inverted_index[term_] = ll
        else:
            postings_list = self.inverted_index[term_]
            postings_list.insert(doc_id_)

    def sort_terms(self):
        """Sort the index by term keys (already implemented)."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        """
        Adds skip pointers to each postings list in the index.

        Rules from project spec:
        - Number of skips = floor(sqrt(L)) where L = postings list length.
        - Skip interval = round(sqrt(L)).
        - If L is a perfect square, use floor(sqrt(L)) - 1 skips.
        """
        for term, plist in self.inverted_index.items():
            L = plist.get_length()
            if L <= 1:
                continue  # no skip pointers needed
            skip_interval = int(round(math.sqrt(L)))
            skip_count = int(math.floor(math.sqrt(L)))
            # perfect square adjustment
            if L == skip_interval * skip_interval:
                skip_count = max(0, skip_count - 1)

            nodes = plist.get_all_nodes()
            for i in range(0, L, skip_interval):
                j = i + skip_interval
                if j < L:
                    nodes[i].skip = nodes[j]

    def calculate_tf_idf(self):
        """
        
        Calculates TF-IDF score for each node (document) in the postings lists.

        Specification:
        - TF = frequency(term in doc) / total tokens in doc
        - IDF = (total number of documents / document frequency of term)
        - TF-IDF = TF * IDF
        """
        # Step 1: find total number of unique documents
        all_docs = set()
        for plist in self.inverted_index.values():
            all_docs.update(plist.get_all_doc_ids())
        total_docs = len(all_docs)

        # Step 2: compute TF-IDF for each term
        for term, plist in self.inverted_index.items():
            df = plist.get_length()
            if df == 0:
                continue
            idf = total_docs / df  # per project spec (no log)

            current = plist.get_head()
            while current:
                doc_id = current.doc_id
                freq = current.freq
                total_tokens = self.doc_token_counts.get(doc_id, 0)
                if total_tokens > 0:
                    tf = freq / total_tokens
                else:
                    tf = 0.0
                current.tf = tf
                current.tfidf = tf * idf
                current = current.next
