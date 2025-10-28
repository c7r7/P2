"""
@author: Sougata Saha
@edited_by: Charan Kumar Raju
Institute: University at Buffalo

Document-At-A-Time (DAAT) Boolean AND processing for Project 2 (CSE 4/535)
Implements:
    • daat_and() – standard DAAT AND
    • daat_and_with_skips() – uses skip pointers
    • rank_by_tfidf() – ranks DAAT results by TF-IDF scores
"""

from math import sqrt


# -------------------------------------------------------------
#  Helper function – pairwise intersection (without skips)
# -------------------------------------------------------------
def intersect_two(plist1, plist2):
    """Return (doc_id list, num comparisons) for two LinkedLists."""
    results = []
    comparisons = 0
    p1 = plist1.get_head()
    p2 = plist2.get_head()

    while p1 and p2:
        comparisons += 1
        if p1.doc_id == p2.doc_id:
            results.append(p1.doc_id)
            p1 = p1.next
            p2 = p2.next
        elif p1.doc_id < p2.doc_id:
            p1 = p1.next
        else:
            p2 = p2.next

    return results, comparisons


# -------------------------------------------------------------
#  Helper function – pairwise intersection (with skips)
# -------------------------------------------------------------
def intersect_two_with_skips(plist1, plist2):
    """Return (doc_id list, num comparisons) using skip pointers."""
    results = []
    comparisons = 0
    p1 = plist1.get_head()
    p2 = plist2.get_head()

    while p1 and p2:
        comparisons += 1
        if p1.doc_id == p2.doc_id:
            results.append(p1.doc_id)
            p1 = p1.next
            p2 = p2.next
        elif p1.doc_id < p2.doc_id:
            # can we skip ahead safely?
            if p1.skip and p1.skip.doc_id <= p2.doc_id:
                comparisons += 1  # compare skip target to p2
                if p1.skip.doc_id <= p2.doc_id:
                    p1 = p1.skip
                else:
                    p1 = p1.next
            else:
                p1 = p1.next
        else:  # p1 > p2
            if p2.skip and p2.skip.doc_id <= p1.doc_id:
                comparisons += 1
                if p2.skip.doc_id <= p1.doc_id:
                    p2 = p2.skip
                else:
                    p2 = p2.next
            else:
                p2 = p2.next

    return results, comparisons


# -------------------------------------------------------------
#  DAAT AND – multi-term Boolean AND (no skips)
# -------------------------------------------------------------
def daat_and(terms, inverted_index):
    """
    Perform Boolean AND across all terms (document-at-a-time).
    Returns: (results list, num_comparisons)
    """
    if not terms:
        return [], 0

    # if any term missing in index → no results
    for t in terms:
        if t not in inverted_index:
            return [], 0

    result_list, total_comparisons = [], 0

    # start with first term’s postings
    current_list = inverted_index[terms[0]]

    # intersect sequentially with each next term
    for i in range(1, len(terms)):
        next_list = inverted_index[terms[i]]
        results, comps = intersect_two(current_list, next_list)
        total_comparisons += comps

        # stop early if no intersection
        if not results:
            return [], total_comparisons

        # build temporary LinkedList-like wrapper for next round
        from linkedlist import LinkedList
        temp = LinkedList()
        for doc in results:
            temp.insert(doc)
        current_list = temp
        result_list = results

    if not result_list:
        result_list = current_list.get_all_doc_ids()

    return result_list, total_comparisons


# -------------------------------------------------------------
#  DAAT AND with skip pointers
# -------------------------------------------------------------
def daat_and_with_skips(terms, inverted_index):
    """Same as daat_and but uses skip pointers for faster traversal."""
    if not terms:
        return [], 0
    for t in terms:
        if t not in inverted_index:
            return [], 0

    result_list, total_comparisons = [], 0
    current_list = inverted_index[terms[0]]

    for i in range(1, len(terms)):
        next_list = inverted_index[terms[i]]
        results, comps = intersect_two_with_skips(current_list, next_list)
        total_comparisons += comps

        if not results:
            return [], total_comparisons

        from linkedlist import LinkedList
        temp = LinkedList()
        for doc in results:
            temp.insert(doc)
        current_list = temp
        result_list = results

    if not result_list:
        result_list = current_list.get_all_doc_ids()

    return result_list, total_comparisons


# -------------------------------------------------------------
#  Rank final DAAT results by TF-IDF
# -------------------------------------------------------------
def rank_by_tfidf(doc_ids, terms, inverted_index):
    """
    Given the result docs from DAAT, compute TF-IDF scores and return:
        [ (doc_id, score) … ] sorted by score desc then doc_id asc
    """
    scores = {}
    for term in terms:
        if term not in inverted_index:
            continue
        plist = inverted_index[term]
        node = plist.get_head()
        while node:
            if node.doc_id in doc_ids:
                scores[node.doc_id] = scores.get(node.doc_id, 0.0) + node.tfidf
            node = node.next

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    # return as list of dicts for JSON friendliness
    return [{"doc_id": d, "score": round(s, 6)} for d, s in ranked]
