"""Bounded ephemeral OpenTela connectivity probe; advertises no model."""

import argparse
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--bootstrap-addr",
        default="/ip4/148.187.108.177/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = "/capstor/store/cscs/swissai/infra01/opentela-share/dev/otela-arm64"
    cfg = args.output / "cfg.yaml"
    cfg.write_text(
        'port: "18092"\np2p:\n  port: "18093"\nqueue:\n  port: "18094"\nloglevel: debug\n'
    )
    cmd = [
        binary,
        "--config",
        str(cfg),
        "--config-dir",
        str(args.output / "config"),
        "start",
        "--ephemeral-identity",
        "--datadir",
        str(args.output / "data"),
        "--tcpport",
        "43906",
        "--udpport",
        "59821",
        "--bootstrap.addr",
        args.bootstrap_addr,
    ]
    tcp = {}
    for ip in ("148.187.108.177", "148.187.108.178"):
        try:
            with socket.create_connection((ip, 43905), timeout=5):
                tcp[ip] = "connected"
        except OSError as error:
            tcp[ip] = str(error)
    result = {
        "argv": cmd,
        "resolved_binary": str(Path(binary).resolve()),
        "tcp": tcp,
        "samples": [],
    }
    with (args.output / "probe.log").open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(6):
                time.sleep(8)
                sample = {"unix": time.time(), "process_exit": proc.poll()}
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18092/v1/dnt/table", timeout=3
                    ) as r:
                        peers = json.load(r)
                    sample["peers"] = [
                        {
                            "id": p.get("id"),
                            "version": p.get("version"),
                            "hostname": p.get("hostname"),
                        }
                        for p in peers.values()
                    ]
                except Exception as error:
                    sample["error"] = str(error)
                result["samples"].append(sample)
                (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
                if proc.poll() is not None:
                    break
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
