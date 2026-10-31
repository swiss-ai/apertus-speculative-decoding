"""Exact and 13-gram Jaccard de-duplication, matching apertus_eagle.overlap_check."""
import hashlib
import re

NGRAM = 13
NEAR_DUP = 0.5


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def shingles(text):
    words = normalize(text)[:2000].split(" ")
    count = max(1, len(words) - NGRAM + 1)
    return [" ".join(words[i:i + NGRAM]) for i in range(count)]


def digest(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


class Index(object):
    def __init__(self):
        self.exact = {}
        self.postings = {}
        self.rows = []

    def add(self, text, row_id):
        key = digest(text)
        self.exact.setdefault(key, row_id)
        grams = shingles(text)
        self.rows.append((row_id, set(grams)))
        index = len(self.rows) - 1
        for gram in set(grams):
            bucket = self.postings.setdefault(gram, [])
            if len(bucket) < 32:
                bucket.append(index)

    def duplicate(self, text):
        key = digest(text)
        if key in self.exact:
            return "exact", self.exact[key]
        grams = set(shingles(text))
        if not grams:
            return None
        seen = set()
        for gram in grams:
            for index in self.postings.get(gram, ()):
                if index in seen:
                    continue
                seen.add(index)
                row_id, other = self.rows[index]
                union = len(grams | other)
                if union and len(grams & other) / float(union) >= NEAR_DUP:
                    return "near", row_id
        return None
