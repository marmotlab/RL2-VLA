#!/usr/bin/env python3
"""Summarize SIMPLER eval logs into per-seed IID/OOD tables (rows=tasks, cols=methods).

Usage:
    python summarize_logs.py [--logs_dir PATH]

    --logs_dir  Directory containing the SEED-*.txt log files.
                Defaults to experiments/logs_ALL_EVALS next to this script.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "logs")

METHOD_ORDER = ["Vanilla", "Rephrase", "Compose-Always", "RL2"]
# Display labels for column headers only; log filenames/matching still use METHOD_ORDER's names.
METHOD_LABEL = {"Compose-Always": "RL2 - Always", "RL2": "RL2 - Adaptive"}

IID_TASKS = ["carrot_on_plate", "put_eggplant_in_basket", "spoon_on_towel", "stack_cube"]
OOD_TASKS = ["orange_juice_on_plate", "spoon_on_towel_google", "tape_measure_in_basket", "toy_dinosaur_on_towel"]

# Log filenames look like:
#   [Rephrase]-SEED-0_simpler_carrot_on_plate-batch-8-5-0-2026_07_25-15_41_18.txt
#   [Compose_Always]-SEED-0_simpler_carrot_on_plate-batch-8-1-5-2026_07_25-17_50_45.txt
#   [RL2]-SEED-0_simpler_carrot_on_plate-batch-8-5-0_8-1-5-alpha-[0.15]-2026_07_26-11_36_58.txt
# NAME_RE strips the trailing timestamp so the remaining regexes can anchor on "$".
NAME_RE = re.compile(r"-\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}\.txt$")
TAG_RE = re.compile(r"^\[([^\]]+)\]")
ALPHA_RE = re.compile(r"alpha-\[([0-9.]+)\]")
TASK_RE = re.compile(r"simpler_([a-z_]+)-batch")
SEED_RE = re.compile(r"SEED-([0-9]+)")


def parse_success_rate(path):
    """Return success rate (%) from the log, or None if missing/truncated."""
    last = None
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if "Total success rate:" in line:
                    last = line
    except OSError:
        return None
    if last is None:
        return None
    parts = last.split()
    try:
        return float(parts[-1]) * 100
    except (ValueError, IndexError):
        return None


def pretty_task(task):
    return task.replace("_", " ").title()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs_dir", default=DEFAULT_LOG_DIR, help="Directory containing the SEED-*.txt log files")
    args = parser.parse_args()
    log_dir = args.logs_dir

    if not os.path.isdir(log_dir):
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # data[seed][task][method] = list of (alpha, value_or_None); one entry per matching
    # log file, so RL2 (which sweeps alpha) naturally accumulates multiple entries.
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    expected = set()  # (task, method) pairs seen in at least one log file, across any seed

    # Pass 1: walk every log file once, parse its name + success rate, and bucket it.
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".txt") or "SEED-" not in fname:
            continue
        stem = NAME_RE.sub("", fname)
        tag_m = TAG_RE.search(stem)
        task_m = TASK_RE.search(stem)
        seed_m = SEED_RE.search(stem)
        if not (tag_m and task_m and seed_m):
            continue
        method = tag_m.group(1)
        task = task_m.group(1)
        seed = seed_m.group(1)
        alpha_m = ALPHA_RE.search(stem)
        alpha = alpha_m.group(1) if alpha_m else ""

        value = parse_success_rate(os.path.join(log_dir, fname))
        data[seed][task][method].append((alpha, value))
        expected.add((task, method))  # this combo was run somewhere, so it's "expected" everywhere

    seeds = sorted(data.keys(), key=lambda s: int(s))

    def cell(seed, task, method):
        if (task, method) not in expected:
            return "-"
        entries = data[seed][task][method]
        valid = [v for _, v in entries if v is not None]
        if not valid:
            return "XX"
        best = max(valid)
        if method == "RL2":
            return f"{best:.1f} (RL2 Top{len(valid)})"
        return f"{best:.1f}"

    # Same logic as cell(), but returns a float (or None) for averaging instead of a display string.
    def numeric(seed, task, method):
        if (task, method) not in expected:
            return None
        entries = data[seed][task][method]
        valid = [v for _, v in entries if v is not None]
        return max(valid) if valid else None

    def print_group(seed, title, tasks):
        headers = ["Task"] + [METHOD_LABEL.get(m, m) for m in METHOD_ORDER]
        rows = []
        for task in tasks:
            row = [pretty_task(task)] + [cell(seed, task, m) for m in METHOD_ORDER]
            rows.append(row)

        avg_row = ["Average"]
        for m in METHOD_ORDER:
            vals = [numeric(seed, task, m) for task in tasks]
            vals = [v for v in vals if v is not None]
            applicable = any((task, m) in expected for task in tasks)
            if not applicable:
                avg_row.append("-")
            elif not vals:
                avg_row.append("XX")
            else:
                avg_row.append(f"{sum(vals) / len(vals):.1f}")
        rows.append(avg_row)

        widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
        print(f"--- {title} ---")
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        print("  ".join("-" * w for w in widths))
        for i, row in enumerate(rows):
            if row[0] == "Average":
                print("  ".join("-" * w for w in widths))
            print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        print()

    # One IID table + one OOD table per seed.
    for seed in seeds:
        print("=" * 60)
        print(f"SEED {seed}")
        print("=" * 60)
        print()
        print_group(seed, "IID", IID_TASKS)
        print_group(seed, "OOD", OOD_TASKS)

    # Average across seeds
    def numeric_avg_across_seeds(task, method):
        if (task, method) not in expected:
            return None, "-"
        vals = [numeric(seed, task, method) for seed in seeds]
        if any(v is None for v in vals):
            # If at least one seed is missing this cell (XX)
            return None, "XX"
        avg = sum(vals) / len(vals)
        return avg, f"{avg:.1f}"

    def print_avg_group(title, tasks):
        headers = ["Task"] + [METHOD_LABEL.get(m, m) for m in METHOD_ORDER]
        rows = []
        for task in tasks:
            row = [pretty_task(task)]
            for m in METHOD_ORDER:
                _, disp = numeric_avg_across_seeds(task, m)
                row.append(disp)
            rows.append(row)

        avg_row = ["Average"]
        for m in METHOD_ORDER:
            vals = []
            applicable = False
            for task in tasks:
                if (task, m) in expected:
                    applicable = True
                    v, _ = numeric_avg_across_seeds(task, m)
                    if v is not None:
                        vals.append(v)
            if not applicable:
                avg_row.append("-")
            elif not vals:
                avg_row.append("XX")
            else:
                avg_row.append(f"{sum(vals) / len(vals):.1f}")
        rows.append(avg_row)

        widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
        print(f"--- {title} ---")
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        print("  ".join("-" * w for w in widths))
        for i, row in enumerate(rows):
            if row[0] == "Average":
                print("  ".join("-" * w for w in widths))
            print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        print()

    if seeds:
        print("=" * 60)
        print(f"AVG: Seed {', '.join(seeds)}")
        print("=" * 60)
        print()
        print_avg_group("IID", IID_TASKS)
        print_avg_group("OOD", OOD_TASKS)


if __name__ == "__main__":
    main()
