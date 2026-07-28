#!/usr/bin/env python3
"""
Restructures collected rollouts into the layout SAFE's data loader expects.

  SRC:
    <run_dir with SEED-X_ in its name>/widowx_task/episode_files

  DST:
    rl2_vla_collected_rollouts_<datetime>/
      widowx_task/
        widowx_task_seedX/
          episode_files

Files are hard-linked (no extra disk space, instant).
"""

import os
import re
import shutil
import time
from pathlib import Path

# SRC = Path("/mnt/hdd/SAFE_ds/training_latents/rollouts/")
# DST = Path("/mnt/hdd/SAFE_ds/training_latents/")
SRC = Path("experiments/")
DST = Path("experiments/restructured_safe")
TOP_FOLDER = f"rl2_vla_collected_rollouts_{time.strftime('%Y%m%d')}"

SEED_RE = re.compile(r"SEED-(\d+)_")


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    total = 0
    for seed_dir in sorted(SRC.iterdir()):
        if not seed_dir.is_dir():
            continue
        m = SEED_RE.search(seed_dir.name)
        if not m:
            print(f"Skipping (no seed prefix): {seed_dir.name}")
            continue
        seed_num = m.group(1)  # e.g. "0", "7", "42"

        for task_dir in sorted(seed_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name  # e.g. widowx_carrot_on_plate

            seed_subfolder = DST / TOP_FOLDER / task_name / f"{task_name}_seed{seed_num}"

            for episode_file in sorted(task_dir.iterdir()):
                if not episode_file.is_file():
                    continue
                dst_file = seed_subfolder / episode_file.name
                hardlink_or_copy(episode_file, dst_file)
                total += 1

    print(f"Done. Linked {total} files to {DST / TOP_FOLDER}")
