#!/usr/bin/env python3
import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a JSON-lines profile log by job names. For each provided job name, "
            "write a new file in the same directory containing only that job in submitted_jobs."
        )
    )
    parser.add_argument("input", help="Path to input profile log (JSON per line)")
    parser.add_argument("jobs", nargs="+", help="Job names to include (space-separated)")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    job_names = [j.strip() for j in args.jobs if j.strip()]
    if not job_names:
        print("Error: no job names provided.", file=sys.stderr)
        sys.exit(2)

    base_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.basename(input_path)
    stem, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".txt"

    # Open one writer per job
    writers = {}
    try:
        for job in job_names:
            out_path = os.path.join(base_dir, f"{stem}__{job}{ext}")
            writers[job] = open(out_path, "w")

        with open(input_path, "r") as fin:
            for line_num, raw in enumerate(fin, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    entry = json.loads(text)
                except Exception as exc:
                    print(
                        f"Warning: failed to parse JSON on line {line_num}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                jobs = entry.get("submitted_jobs", [])
                for job in job_names:
                    filtered = dict(entry)
                    if isinstance(jobs, list):
                        filtered_jobs = [j for j in jobs if isinstance(j, dict) and j.get("name") == job]
                    else:
                        filtered_jobs = []
                    filtered["submitted_jobs"] = filtered_jobs
                    writers[job].write(json.dumps(filtered) + "\n")
    finally:
        for fp in writers.values():
            try:
                fp.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()


