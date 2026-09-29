# Client-limited above concurrency 100

This sweep ran with httpx's default connection pool (100 connections). At
C=128 and C=256 the server never had more than 99 running requests and 0
waiting, with the KV cache at most 62% full, while client-side TTFT grew to
18 s: requests queued in the client, not the server. Levels up to C=64 are
valid. Fixed in the harness (no pool cap); superseded by the rerun.
