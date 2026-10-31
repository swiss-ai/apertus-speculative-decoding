"""Cache large GHCR layers with bounded, resumable HTTP/1.1 range requests.

The compressed OCI checksum is verified before reproducing Enroot's gzip to
zstd cache conversion. Existing cache entries are never replaced.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def prefetch(manifest_path, parts_root, cache):
    repository = "eth-cscs/alps-extended-images/vllm-rocm"
    with urllib.request.urlopen("https://ghcr.io/token?scope=repository:" + repository + ":pull&service=ghcr.io", timeout=60) as response:
        token = json.load(response)["token"]
    manifest = json.loads(manifest_path.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    for layer in manifest["layers"]:
        if layer["size"] < 1024 ** 3:
            continue
        digest = layer["digest"].split(":", 1)[1]
        destination = cache / digest
        if destination.exists():
            print("Layer already cached: " + digest, flush=True)
            continue
        assert layer["mediaType"].endswith("tar.gzip")
        parts = parts_root / digest
        parts.mkdir(parents=True, exist_ok=True)
        url = "https://ghcr.io/v2/" + repository + "/blobs/sha256:" + digest
        chunk_size = 256 * 1024 ** 2
        ranges = [(i, start, min(start + chunk_size, layer["size"]) - 1)
                  for i, start in enumerate(range(0, layer["size"], chunk_size))]

        def download(item):
            index, start, end = item
            path = parts / ("{:04d}.gz-part".format(index))
            expected = end - start + 1
            if path.exists() and path.stat().st_size == expected:
                return path
            for attempt in range(4):
                temporary = path.with_suffix(".partial")
                try:
                    request = urllib.request.Request(url, headers={
                        "Authorization": "Bearer " + token,
                        "Range": "bytes={}-{}".format(start, end),
                    })
                    with urllib.request.urlopen(request, timeout=60) as response:
                        assert response.status == 206, response.status
                        assert response.headers["Content-Range"] == "bytes {}-{}/{}".format(start, end, layer["size"])
                        with temporary.open("wb") as output:
                            while True:
                                block = response.read(1024 ** 2)
                                if not block:
                                    break
                                output.write(block)
                    assert temporary.stat().st_size == expected
                    temporary.rename(path)
                    print("Downloaded piece {}/{}".format(index + 1, len(ranges)), flush=True)
                    return path
                except Exception as error:
                    print("Retry piece {}: {}".format(index + 1, type(error).__name__), flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)

        with ThreadPoolExecutor(max_workers=8) as executor:
            downloaded = list(executor.map(download, ranges))
        checksum = hashlib.sha256()
        for path in downloaded:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 ** 2), b""):
                    checksum.update(block)
        assert checksum.hexdigest() == digest, "OCI layer checksum mismatch"
        print("Verified compressed layer SHA-256: " + digest, flush=True)
        temporary = cache / (digest + ".prefetch-" + str(os.getpid()))
        unzip = subprocess.Popen(["pigz", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        compress = subprocess.Popen(["zstd", "-T8", "-q", "-1", "-o", str(temporary)], stdin=unzip.stdout)
        unzip.stdout.close()
        try:
            for path in downloaded:
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(4 * 1024 ** 2), b""):
                        unzip.stdin.write(block)
        finally:
            unzip.stdin.close()
        unzip_rc = unzip.wait()
        compress_rc = compress.wait()
        assert unzip_rc == 0 and compress_rc == 0, (unzip_rc, compress_rc)
        os.chmod(str(temporary), 0o640)
        try:
            os.link(str(temporary), str(destination))
        except FileExistsError:
            pass
        temporary.unlink()
        print("Cached verified layer: " + digest, flush=True)


if __name__ == "__main__":
    prefetch(*[Path(value) for value in sys.argv[1:]])
