#!/usr/bin/env python3
"""fetch_github.py <owner/repo> <commit> <dest> [exclude-regex]: download the code files of a GitHub commit one by one
(git trees API + raw.githubusercontent.com) into <dest>. Used by self-host setup.sh scripts instead of the codeload
tarball, which would also download every media asset of the repository (slow behind a proxy). Uses https_proxy if set."""
import json
import os
import re
import sys
import urllib.request

repo, commit, dest = sys.argv[1:4]
skip = re.compile(sys.argv[4] if len(sys.argv) > 4 else r"\.(mp4|gif|pdf|png|jpg|jpeg|webp|wav|mp3|mov)$", re.I)


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            if i == tries - 1:
                raise
            print("retry", url, exc, file=sys.stderr)


tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
assert not tree.get("truncated"), "tree truncated"
n = 0
for t in tree["tree"]:
    if t["type"] != "blob" or skip.search(t["path"]):
        continue
    out = os.path.join(dest, t["path"])
    if os.path.exists(out) and os.path.getsize(out) == t.get("size", -1):
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{t['path']}")
    with open(out + ".part", "wb") as f:
        f.write(data)
    os.replace(out + ".part", out)
    n += 1
print(f"{repo}@{commit[:12]}: {n} files fetched into {dest}")
