"""Benchmarks the two-stage planner (course generation + hybrid-A* +
arm-angle search) across a set of random seeds. No PyBullet/GUI needed --
course generation and both planning stages are pure Python/math. Prints a
table of obstacle count, regeneration attempts, path length, and wall-clock
time per seed, plus summary stats.

This is what produced the preliminary-results table in
project_status/project_status.tex; rerun it any time that table needs
fresh or additional numbers.

Run:
    python planner_benchmark.py                  # the 10 seeds used in the report
    python planner_benchmark.py --n 25            # 25 fresh random seeds
    python planner_benchmark.py --seeds 1 2 3     # specific seeds
    python planner_benchmark.py --latex           # also print LaTeX tabular rows
"""

import argparse
import math
import random
import time

from config import COURSE_LENGTH
from planning import generate_solvable_course

# The seeds used in the project_status.tex preliminary-results table.
REPORT_SEEDS = [1, 2, 3, 42, 1000, 999999, 7, 55, 2024, 31415]


def path_length(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def run_one(seed):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    obstacles, path, _phis, attempts = generate_solvable_course(rng)
    elapsed = time.perf_counter() - t0
    return {
        "seed": seed,
        "obstacles": len(obstacles),
        "attempts": attempts,
        "path_len": path_length(path),
        "time_s": elapsed,
    }


def print_table(rows):
    print(f"{'seed':>8} {'obst':>5} {'attempts':>9} {'path_len':>9} {'time_s':>8}")
    for r in rows:
        print(f"{r['seed']:>8} {r['obstacles']:>5} {r['attempts']:>9} "
              f"{r['path_len']:>9.2f} {r['time_s']:>8.2f}")


def print_summary(rows):
    attempts = [r["attempts"] for r in rows]
    times = [r["time_s"] for r in rows]
    overhead = [(r["path_len"] - COURSE_LENGTH) / COURSE_LENGTH * 100 for r in rows]
    print()
    print(f"mean attempts={sum(attempts) / len(attempts):.2f}  max attempts={max(attempts)}")
    print(f"mean time={sum(times) / len(times):.3f}s  max time={max(times):.3f}s")
    print(f"path overhead: min={min(overhead):.1f}%  max={max(overhead):.1f}%  "
          f"mean={sum(overhead) / len(overhead):.1f}%")


def print_latex_rows(rows):
    print("\n% LaTeX tabular rows (seed & obst. & attempts & path (m) & time (s)):")
    for r in rows:
        print(f"{r['seed']:<7} & {r['obstacles']} & {r['attempts']} & "
              f"{r['path_len']:.2f} & {r['time_s']:.2f} \\\\")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Explicit seeds to run (default: the seeds used in the status report)")
    parser.add_argument("--n", type=int, default=None,
                         help="Run N fresh random seeds instead of --seeds/the report defaults")
    parser.add_argument("--latex", action="store_true", help="Also print rows as LaTeX tabular body")
    args = parser.parse_args()

    if args.n:
        seeds = [random.randrange(2 ** 31) for _ in range(args.n)]
    else:
        seeds = args.seeds if args.seeds else REPORT_SEEDS

    rows = [run_one(seed) for seed in seeds]
    print_table(rows)
    print_summary(rows)
    if args.latex:
        print_latex_rows(rows)


if __name__ == "__main__":
    main()
