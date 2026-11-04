import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/tree"))

from apertus_tree.trie import NgramIndex, Trie  # noqa: E402


def test_ngram_branches_longest_then_most_recent():
    #       0  1  2  3  4  5  6  7  8  9
    ids = [5, 6, 7, 8, 5, 6, 9, 1, 5, 6]
    index = NgramIndex(ids, max_n=4, min_n=1)
    branches, longest = index.branches(9, limit=8, depth=3)
    # Suffix "5 6" matched at ends 5 (-> 9 1 5) and 1 (-> 7 8 5); then "6" alone
    # adds nothing new; the unigram history of 6 is the same two positions.
    assert longest == 2
    assert branches[:2] == [[9, 1, 5], [7, 8, 5]]
    # Continuations never read past the anchor.
    branches, _ = index.branches(5, depth=7)
    assert branches[0] == [7, 8, 5, 6]


def test_ngram_without_match():
    index = NgramIndex([1, 2, 3, 4])
    assert index.branches(3) == ([], 0)


def test_trie_merges_shared_prefixes_and_keeps_best_rank():
    trie = Trie()
    assert trie.add_path([1, 2, 3], flag=1) == [0, 1, 2]
    assert trie.add_path([1, 2, 4], flag=2, ng=3) == [0, 1, 3]
    trie.add_path([1, 5], flag=2, ng=1)
    assert trie.parent == [-1, 0, 1, 1, 0]
    assert trie.depth == [1, 2, 3, 3, 2]
    assert trie.flags == [3, 3, 1, 2, 2]
    assert trie.ng == [1, 3, -1, 3, 1]
    assert trie.path(3) == [1, 2, 4]
