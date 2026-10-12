#!/usr/bin/env python3
"""Export verified full results to GitHub without committing private features.

Default is read-only.  --export copies only sanitized aggregate metrics,
protocol and verification files. --publish requires --confirm-publish.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/patch_benchmark_v1/reports/full/main"
PUBLIC = ROOT / "benchmark_reports/patch_retrieval_fewshot_v1"
SOURCE_FILES = ("summary.csv", "comparison.csv", "summary.json",
                "comparison.json", "verification.json")
THREE_METHOD_FILES = ("summary.csv", "summary.json", "per_dataset.csv", "verification.json")


def verified():
    report = REPORTS / "verification.json"
    if not report.is_file():
        raise FileNotFoundError("Full 150-result benchmark has not completed")
    status = json.loads(report.read_text())
    if not status["final_complete"] or status["completed_jobs"] != 150 or status["missing"]:
        raise RuntimeError("Cannot export incomplete evaluation")
    for filename in SOURCE_FILES:
        if not (REPORTS / filename).is_file():
            raise FileNotFoundError(REPORTS / filename)
    rows = json.loads((REPORTS / "comparison.json").read_text())
    # 5 epochs * 15 datasets * (6 retrieval + 9 few-shot) comparisons.
    if len(rows) != 1125:
        raise RuntimeError("Cannot export incomplete per-dataset comparison")
    if any("feature_file" in r or "checkpoint_path" in r or
           "relative_path" in r or "support_subsets" in r for r in rows):
        raise RuntimeError("Private sample or checkpoint paths detected in public result")
    report3 = REPORTS / "three_methods" / "verification.json"
    if not report3.is_file():
        raise FileNotFoundError("Missing three-method verification")
    r3 = json.loads(report3.read_text())
    if (r3.get("total_method_checkpoint_dataset_results") != 210 or
            not r3.get("aggregate_independently_verified") or
            not r3.get("all_input_identity_checked")):
        raise RuntimeError("Three-method benchmark is incomplete")
    for name in THREE_METHOD_FILES:
        if not (REPORTS / "three_methods" / name).is_file():
            raise FileNotFoundError(name)
    return status


def export():
    verified()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        shutil.copy2(REPORTS / filename, PUBLIC / filename)
    (PUBLIC / "three_methods").mkdir(parents=True, exist_ok=True)
    for name in THREE_METHOD_FILES:
        shutil.copy2(REPORTS / "three_methods" / name,
                     PUBLIC / "three_methods" / name)
    internal_protocol = (ROOT / "docs/PATCH_BENCHMARK_PROTOCOL.md").read_text()
    # Remove private server hostnames and storage paths from the public report.
    public_protocol = internal_protocol.split("## Three-server parallel scheduler")[0]
    if any(text in public_protocol for text in ("/raid1/", "/home/", ".it.usyd.edu.au", ".research.sydney.edu.au")):
        raise RuntimeError("Internal machine identifier in public protocol")
    (PUBLIC / "PROTOCOL.md").write_text(public_protocol)
    print("EXPORTED", PUBLIC)
    return PUBLIC


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    return p.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true",
                        help="Write public-summary-only report package; no network")
    parser.add_argument("--publish", action="store_true",
                        help="Commit only the verified report package and push to GitHub")
    parser.add_argument("--confirm-publish", action="store_true")
    args = parser.parse_args()
    if args.publish and not args.confirm_publish:
        parser.error("Publishing to GitHub requires --confirm-publish")
    if not args.publish and not args.export:
        print("NO ACTION: add --export, or --publish --confirm-publish after full run.")
        return
    export()
    if args.publish:
        rel = PUBLIC.relative_to(ROOT).as_posix()
        git("add", "-f", "--", rel)
        git("commit", "--only", "-m",
            "Publish verified LeJEPA K64 K8 retrieval and few-shot results",
            "--", rel)
        print(git("push", "origin", "HEAD"))
        print("PUBLISHED", rel)


if __name__ == "__main__":
    main()
