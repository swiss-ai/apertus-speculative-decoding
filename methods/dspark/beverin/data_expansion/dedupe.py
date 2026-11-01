"""Exact and 13-gram Jaccard de-duplication, matching apertus_eagle.overlap_check."""
import hashlib
import re
import zlib

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


class BoundedIndex(object):
    """Exact hashes plus a MinHash estimate of 13-gram Jaccard.

    The full shingle posting list for 1.5M open-perfectblend rows does not fit
    in one node's memory. Eight 32-bit minhashes estimate Jaccard, and a hit at
    or above 0.5 is reported as a near duplicate. Exact normalized matches are
    unchanged.
    """

    def __init__(self, perms=8, bands=4):
        if perms % bands:
            raise ValueError("perms must divide into bands")
        self.perms = perms
        self.bands = bands
        self.rows_per_band = perms // bands
        self.salts = tuple((0x9E3779B1 * (i + 1)) & 0xFFFFFFFF for i in range(perms))
        self.exact = {}
        self.ids = []
        self.signatures = []
        self.buckets = {}

    def _signature(self, text):
        words = normalize(text)[:2000].split(" ")
        count = max(1, len(words) - NGRAM + 1)
        signature = [0xFFFFFFFF] * self.perms
        salts = self.salts
        for offset in range(count):
            base = zlib.crc32(" ".join(words[offset:offset + NGRAM]).encode("utf-8")) & 0xFFFFFFFF
            for perm, salt in enumerate(salts):
                mixed = (base * salt) & 0xFFFFFFFF
                if mixed < signature[perm]:
                    signature[perm] = mixed
        return tuple(signature)

    def add(self, text, row_id):
        key = digest(text)
        self.exact.setdefault(key, row_id)
        signature = self._signature(text)
        index = len(self.ids)
        self.ids.append(row_id)
        self.signatures.append(signature)
        width = self.rows_per_band
        for band in range(self.bands):
            part = signature[band * width:(band + 1) * width]
            bucket = self.buckets.setdefault((band, part), [])
            if len(bucket) < 32:
                bucket.append(index)

    def duplicate(self, text):
        key = digest(text)
        if key in self.exact:
            return "exact", self.exact[key]
        signature = self._signature(text)
        seen = set()
        width = self.rows_per_band
        for band in range(self.bands):
            part = signature[band * width:(band + 1) * width]
            for index in self.buckets.get((band, part), ()):
                if index in seen:
                    continue
                seen.add(index)
                other = self.signatures[index]
                agree = sum(1 for left, right in zip(signature, other) if left == right)
                if agree / float(self.perms) >= NEAR_DUP:
                    return "near", self.ids[index]
        return None
