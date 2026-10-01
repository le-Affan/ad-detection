"""Refuse test-split access unless analysis/fusion_v3/freeze.md is committed and unmodified."""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def assert_frozen():
    rel = "analysis/fusion_v3/freeze.md"
    log = subprocess.run(["git", "-C", ROOT, "log", "--format=%H", "-1", "--", rel], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", rel], capture_output=True, text=True)
    assert log.stdout.strip() and not dirty.stdout.strip(), "test access requires a committed, unmodified freeze.md"
    return log.stdout.strip()
