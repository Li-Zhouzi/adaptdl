#!/usr/bin/env python3
"""Filter monitor logs for specific deepspeech2 jobs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Optional, Tuple
from pprint import pprint


TARGET_JOBS = {"deepspeech2-60", "deepspeech2-73", "deepspeech2-22", "deepspeech2-155"}
DROP_JOB_FIELDS = {"submission_time", "completion_time", "grad_params"}


def process_line(
    line: str,
    job_pairs: Optional[DefaultDict[str, Dict[Tuple[int, Optional[int]], float]]] = None,
) -> Optional[str]:
    """Return a JSON string for lines containing the target jobs, else None."""

    if not line.strip():
        return None

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None

    submitted_jobs = record.get("submitted_jobs") or []

    filtered_jobs = []
    timestamp = record.get("timestamp")
    for job in submitted_jobs:
        name = job.get("name")
        if name in TARGET_JOBS:
            filtered_jobs.append({k: v for k, v in job.items() if k not in DROP_JOB_FIELDS})
            if job_pairs is not None and timestamp is not None:
                allocation = job.get("allocation") or []
                batch_size = job.get("batch_size")
                pair = (len(allocation), batch_size)
                job_pairs[name].setdefault(pair, timestamp)

    if not filtered_jobs:
        return None

    filtered_record = dict(record)
    filtered_record["submitted_jobs"] = filtered_jobs
    return json.dumps(filtered_record)


def process_log(
    lines: Iterable[str],
    job_pairs: Optional[DefaultDict[str, Dict[Tuple[int, Optional[int]], float]]] = None,
) -> Iterable[str]:
    for line in lines:
        processed = process_line(line, job_pairs)
        if processed is not None:
            yield processed + "\n"


def summarize_job_pairs(
    job_pairs: DefaultDict[str, Dict[Tuple[int, Optional[int]], float]]
) -> dict[str, list[Tuple[int, Optional[int]]]]:
    summary: dict[str, list[Tuple[int, Optional[int]]]] = {}
    for job, pairs in job_pairs.items():
        ordered = sorted(pairs.items(), key=lambda item: item[1])
        summary[job] = [pair for pair, _ in ordered]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the monitor log file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("test.txt"),
        help="Path to write the filtered log (default: test.txt).",
    )
    args = parser.parse_args()

    job_pairs: DefaultDict[str, Dict[Tuple[int, Optional[int]], float]] = defaultdict(dict)

    with args.input_path.open("r", encoding="utf-8") as infile, args.output.open(
        "w", encoding="utf-8"
    ) as outfile:
        for processed_line in process_log(infile, job_pairs):
            outfile.write(processed_line)

    summary = summarize_job_pairs(job_pairs)
    pprint(summary)


if __name__ == "__main__":
    main()


