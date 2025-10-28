"""
@author: Sougata Saha
@edited_by: Charan Kumar Raju
Institute: University at Buffalo

LinkedList implementation for Project 2 (CSE 4/535)
Used to represent postings lists in the inverted index.
"""

class Node:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.freq = 1           # frequency of term in this document
        self.tf = 0.0           # term frequency (normalized)
        self.tfidf = 0.0        # tf-idf score
        self.next = None        # pointer to next node
        self.skip = None        # skip pointer


class LinkedList:
    def __init__(self):
        self.head = None
        self.length = 0

    # ------------------------------------------------------------
    # Insert doc_id in sorted order; increment freq if already present
    # ------------------------------------------------------------
    def insert(self, doc_id):
        # Empty list → create first node
        if self.head is None:
            self.head = Node(doc_id)
            self.length = 1
            return

        # If new doc_id is smaller → prepend
        if doc_id < self.head.doc_id:
            new_node = Node(doc_id)
            new_node.next = self.head
            self.head = new_node
            self.length += 1
            return

        # Traverse the list to find position
        prev = None
        cur = self.head
        while cur and cur.doc_id < doc_id:
            prev = cur
            cur = cur.next

        # Case 1: found existing doc_id → increment frequency
        if cur and cur.doc_id == doc_id:
            cur.freq += 1
            return

        # Case 2: insert between prev and cur (sorted order)
        new_node = Node(doc_id)
        new_node.next = cur
        if prev:
            prev.next = new_node
        self.length += 1

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------
    def get_length(self):
        return self.length

    def get_head(self):
        return self.head

    def get_all_nodes(self):
        nodes = []
        cur = self.head
        while cur:
            nodes.append(cur)
            cur = cur.next
        return nodes

    def get_all_doc_ids(self):
        ids = []
        cur = self.head
        while cur:
            ids.append(cur.doc_id)
            cur = cur.next
        return ids

    # ------------------------------------------------------------
    # NEW method (fixes your AttributeError)
    # ------------------------------------------------------------
    def get_skip_doc_ids(self):
        """
        Returns a list of document IDs that each skip pointer jumps to.
        Example: [4, 7, 10]
        """
        skip_ids = []
        cur = self.head
        while cur:
            if cur.skip:
                skip_ids.append(cur.skip.doc_id)
            cur = cur.next
        return skip_ids
