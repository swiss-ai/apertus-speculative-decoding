"""Save the first N rows of a prepared DSpark corpus (with its token_freq.pt).

    python3 data_slice.py SOURCE DEST N
"""
import shutil
import sys
from pathlib import Path

from datasets import load_from_disk

source, dest, rows = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if dest.exists():
    sys.exit("refusing to overwrite %s" % dest)
data = load_from_disk(str(source)).select(range(rows))
data.save_to_disk(str(dest))
shutil.copy2(str(source / "token_freq.pt"), str(dest / "token_freq.pt"))
print(data)
