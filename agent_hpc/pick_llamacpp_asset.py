#!/usr/bin/env python3
"""Pick the Linux CUDA x64 asset out of a llama.cpp GitHub release payload.

Used by agent_hpc/pbs/run_llamacpp_agent_server.pbs. Kept as a file rather than
inlined in the job so the quoting stays sane and it can be tested on its own:

    python3 agent_hpc/pick_llamacpp_asset.py <(curl -s .../releases/latest)

Prints the download URL, or exits 1 if the release has no such asset.
"""

import json
import re
import sys

PATTERN = re.compile(r"ubuntu.*cuda.*x64.*\.zip$", re.IGNORECASE)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: pick_llamacpp_asset.py <release.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        release = json.load(handle)
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if PATTERN.search(name):
            print(asset["browser_download_url"])
            return 0
    print("no ubuntu/cuda/x64 zip in this release", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
