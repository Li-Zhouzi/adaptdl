import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FILENAME_PATTERN = re.compile(r"^restart(\d+)-pod(\d+)(?:\.txt)?$")
STEP_LINE_PATTERN = re.compile(
    r"^\[TIMING\]\s+\d+\.\s+([^:]+):\s+duration=([0-9.]+)s"
)


def parse_steps_from_lines(lines: List[str]) -> Dict[str, float]:
    steps: Dict[str, float] = {}
    summary_start_indices: List[int] = [
        idx for idx, line in enumerate(lines)
        if line.strip().startswith("[TIMING] Pre-training steps summary:")
    ]

    if not summary_start_indices:
        return steps

    # Use the last summary block if multiple are present
    start_idx = summary_start_indices[-1] + 1
    for i in range(start_idx, len(lines)):
        line = lines[i].rstrip("\n")
        match = STEP_LINE_PATTERN.match(line.strip())
        if not match:
            # Stop once we leave the numbered timing steps block
            if steps:
                break
            else:
                continue

        step_name = match.group(1).strip()
        try:
            duration_seconds = float(match.group(2))
        except ValueError:
            # Skip malformed duration lines gracefully
            continue

        steps[step_name] = duration_seconds

    return steps


def parse_log_file(file_path: Path) -> Dict[str, float]:
    try:
        contents = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    lines = contents.splitlines()
    return parse_steps_from_lines(lines)


def scan_directory(directory: Path) -> Dict[int, Dict[int, Dict[str, float]]]:
    results: Dict[int, Dict[int, Dict[str, float]]] = {}
    if not directory.exists() or not directory.is_dir():
        return results

    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = FILENAME_PATTERN.match(entry.name)
        if not match:
            continue

        restart_idx = int(match.group(1))
        pod_idx = int(match.group(2))
        steps = parse_log_file(entry)
        if not steps:
            continue

        if restart_idx not in results:
            results[restart_idx] = {}
        results[restart_idx][pod_idx] = steps

    return results


def main() -> None:
    # Default directory per request; can be overridden by CLI arg
    default_dir = Path("./experiment_results/1031-restart-test")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    directory = dir_arg.resolve()

    results = scan_directory(directory)

    # For now, just print the dictionary as JSON for quick inspection
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


