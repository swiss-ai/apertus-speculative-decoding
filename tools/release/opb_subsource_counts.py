"""Count non-permissive UltraInteract subparts inside open-perfectblend.

Reads the six parquet files of ``mlabonne/open-perfectblend`` at the revision
the 8B drafters were trained on, over HTTPS with DuckDB (nothing is stored),
and counts rows whose first message matches each subpart's prompt template.
Used by docs/release/license-audit-8b-drafters.md.

    uv run --no-project --with duckdb python -I tools/release/opb_subsource_counts.py
"""

import json

import duckdb

REVISION = "af60f3c18201652a83a93f46fcfee1b646ba3df7"
URLS = [
    "https://huggingface.co/datasets/mlabonne/open-perfectblend/resolve/"
    f"{REVISION}/data/train-0000{i}-of-00006.parquet"
    for i in range(6)
]
PATTERNS = {
    "reclor_template": (
        "%Given the context and corresponding question, choose the correct answer from the options%"
    ),
    "tabmwp_template": "%Given the following table:%Table title:%",
    "hotpotqa_strategyqa_wiki_tool": "%wikipedia_search(query: str)%",
    "wikitableq_template": "%Given the following table (only the first 5 rows are shown)%",
}


def main() -> None:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    counts = ",\n  ".join(
        f"count(*) FILTER (WHERE p LIKE '{pattern}') AS {name}"
        for name, pattern in PATTERNS.items()
    )
    query = f"""
    WITH t AS (SELECT source, conversations[1].value AS p FROM read_parquet({URLS!r}))
    SELECT source, count(*) AS rows,
      {counts}
    FROM t GROUP BY source ORDER BY rows DESC
    """
    names = ["source", "rows", *PATTERNS]
    for row in con.execute(query).fetchall():
        print(json.dumps({"revision": REVISION, **dict(zip(names, row, strict=True))}))


if __name__ == "__main__":
    main()
