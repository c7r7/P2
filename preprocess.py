'''
@author: Sougata Saha
Institute: University at Buffalo
'''

import collections
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


class Preprocessor:
    def __init__(self):
        """Initialize stopword list and stemmer."""
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def get_doc_id(self, doc):
        """Splits each line of the document into doc_id & text."""
        arr = doc.split("\t")
        return int(arr[0]), arr[1]

    def tokenizer(self, text):
        """
        Preprocess and tokenize text according to the Project 2 specification.

        Steps:
        1. Convert text to lowercase.
        2. Replace all non-alphanumeric characters with a space.
        3. Collapse multiple spaces into one.
        4. Tokenize by whitespace.
        5. Remove stopwords.
        6. Apply Porter stemming.

        Returns:
            List[str]: list of processed tokens.
        """
        # 1. Lowercase
        text = text.lower()

        # 2. Remove all non-alphanumeric chars (keep letters, digits, and spaces)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # 3. Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. Tokenize by whitespace
        tokens = text.split()

        # 5. Remove stopwords
        filtered_tokens = [t for t in tokens if t not in self.stop_words]

        # 6. Apply Porter Stemmer
        stemmed_tokens = [self.ps.stem(t) for t in filtered_tokens]

        return stemmed_tokens
