"""Primary ids of Open-PerfectBlend conversations with no row in the training split.

    python3 tools/opb-clean-prompts.py CORPUS.jsonl VAL_IDS.txt OUT.txt

The DSpark split (val_ids.txt) holds out rows, and a conversation has one row
per regenerated turn (<primary_id>_gen<k>); a conversation is clean when every
one of its rows is held out. Reads only the "id" field of each corpus line.
"""

import sys
from collections import Counter

corpus, val_path, out = sys.argv[1:4]
with open(val_path) as handle:
    val = {line.strip() for line in handle if line.strip()}
total: Counter[str] = Counter()
held: Counter[str] = Counter()
with open(corpus, "rb") as handle:
    for line in handle:
        start = line.find(b'"id": "') + 7
        row_id = line[start : line.find(b'"', start)].decode()
        prompt = row_id.rsplit("_gen", 1)[0]
        total[prompt] += 1
        held[prompt] += row_id in val
clean = sorted(p for p, n in total.items() if held[p] == n)
with open(out, "w") as handle:
    handle.write("".join(p + "\n" for p in clean))
print(f"{len(total)} conversations, {sum(total.values())} rows, {len(clean)} fully held out")
