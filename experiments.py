"""Batch experiments behind the final report's results tables. Every run is
headless and deterministic per seed; each experiment writes its raw rows
to results/<name>.json and prints a summary plus LaTeX table rows.

  planners  Live planners (dwa, dstar, hybrid) and the oracle (offline
            planner with full knowledge) on the same roaming courses.
  redirect  Objective 4 ablation: the full planner vs. a no-redirect
            baseline (ball rigidly beside the chassis) on the same raw
            courses, with why each failure happened.
  density   Objective 5: completion, collisions, path overhead, and
            redirect frequency vs. obstacle count, from an obstacle-free
            baseline (0 obstacles) up.
  robustness  The live planner with roaming obstacles' random turns off
            and on (which its prediction can't foresee), for both
            predictors: how much it degrades when its motion model is wrong.

Run:
    python experiments.py planners [--seeds 30]
    python experiments.py redirect [--courses 60]
    python experiments.py density  [--seeds 20]
    python experiments.py robustness [--seeds 60]
"""

import argparse
import contextlib
import io
import json
import math
import os
import random
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _run_quiet(job):
    """One headless simulation run in a worker process. `overrides` patches
    course-generation parameters on the obstacles module (the course cache
    keys on them, so overridden courses never collide with default ones)."""
    seed, planner, motion, overrides = job
    import importlib
    import obstacles
    import simulation
    for name, value in overrides.items():
        module, _, attr = name.rpartition(".")   # "live_planning.PREDICTION", or a bare obstacles attribute
        setattr(importlib.import_module(module) if module else obstacles, attr, value)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            result = simulation.run(seed, planner, motion, headless=True, verbose=False)
        except RuntimeError as e:   # no acceptable course for this seed
            result = {"seed": seed, "planner": planner, "complete": False, "error": str(e)}
    result["overrides"] = {k: list(v) if isinstance(v, tuple) else v for k, v in overrides.items()}
    return result


def _plan_quiet(job):
    """Objective 4 ablation on one raw course: plan it with and without
    redirection, recording success, failure stage, and path length."""
    seed, = job
    from obstacles import generate_course
    from planning import plan_course
    obstacles = generate_course(random.Random(seed), "oscillate")
    row = {"seed": seed, "obstacles": len(obstacles), "moving": sum(o.moving for o in obstacles)}
    for label, redirect in (("full", True), ("no_redirect", False)):
        path, _phis, reason = plan_course(obstacles, redirect=redirect)
        row[label] = reason
        row[f"{label}_len"] = (sum(math.dist(a[:2], b[:2]) for a, b in zip(path, path[1:]))
                               if path else None)
    return row


def _pool_map(fn, jobs):
    with Pool(WORKERS) as pool:
        return pool.map(fn, jobs)


def _save(name, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"raw rows: {path}")


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def summarize_runs(rows):
    done = [r for r in rows if r.get("complete")]
    ran = [r for r in rows if "time_s" in r]
    hits = [r.get("body_hits", 0) + r.get("ball_hits", 0) for r in ran]
    return {
        "n": len(rows), "done": len(done), "no_plan": sum(1 for r in rows if r.get("no_plan")),
        "collisions": sum(hits), "runs_with_collision": sum(1 for h in hits if h),
        "time_s": _mean(r["time_s"] for r in done),
        "overhead_pct": _mean(100 * (r["body_path_len"] - 6.0) / 6.0 for r in done),
        "overhead_median_pct": _median(100 * (r["body_path_len"] - 6.0) / 6.0 for r in done),
        "replan_ms": _mean(r["replan_ms_mean"] for r in ran if "replan_ms_mean" in r),
        "replan_ms_max": max((r["replan_ms_max"] for r in ran if "replan_ms_max" in r), default=float("nan")),
        "obstacles": _mean(r["obstacles"] for r in ran),
        "moving": _mean(r["moving"] for r in ran),
        "energy_other_pct": _mean(100 * abs(r["other"]) / r["paddle_work"] for r in ran if r.get("paddle_work")),
        "binding_obstacle": _mean(r.get("binding_obstacle", 0.0) for r in ran if "replans" in r),
        "binding_turn_rate": _mean(r.get("binding_turn_rate", 0.0) for r in ran if "replans" in r),
        "binding_ball_speed": _mean(r.get("binding_ball_speed", 0.0) for r in ran if "replans" in r),
        "local_avoid": _mean(r.get("binding_local_avoid", 0.0) for r in ran if "replans" in r),
        "redirect_forced": _mean(r["redirect_forced_frac"] for r in ran if "redirect_forced_frac" in r),
    }


def exp_planners(n_seeds):
    planners = ["dwa", "dstar", "hybrid", "oracle"]
    rows = _pool_map(_run_quiet, [(s, p, "roam", {}) for p in planners for s in range(n_seeds)])
    _save("planners", rows)
    print(f"\n{'planner':8s} done  no-plan  coll  runs-w/coll  time(s)  overhead  replan mean/max (ms)")
    for p in planners:
        s = summarize_runs([r for r in rows if r["planner"] == p])
        print(f"{p:8s} {s['done']:2d}/{s['n']:<2d} {s['no_plan']:6d} {s['collisions']:5d} {s['runs_with_collision']:8d}"
              f"      {s['time_s']:6.1f}  {s['overhead_pct']:6.1f}%   {s['replan_ms']:.0f} / {s['replan_ms_max']:.0f}")
    print(f"\ncourses: {summarize_runs(rows)['obstacles']:.1f} obstacles, {summarize_runs(rows)['moving']:.1f} moving on average")
    print("\n% LaTeX rows: planner & done & collisions & time & replan")
    for p in planners:
        s = summarize_runs([r for r in rows if r["planner"] == p])
        replan = "---" if p == "oracle" else f"{s['replan_ms']:.0f} / {s['replan_ms_max']:.0f}"
        print(f"{p} & {s['done']}/{s['n']} & {s['collisions']} & {s['time_s']:.1f} & {replan} \\\\")


def exp_redirect(n_courses):
    rows = _pool_map(_plan_quiet, [(s,) for s in range(n_courses)])
    _save("redirect", rows)
    full_ok = [r for r in rows if r["full"] == "ok"]
    base_ok = [r for r in rows if r["no_redirect"] == "ok"]
    rescued = [r for r in rows if r["full"] == "ok" and r["no_redirect"] != "ok"]
    both = [r for r in rows if r["full"] == "ok" and r["no_redirect"] == "ok"]
    print(f"\n{len(rows)} raw courses ({_mean(r['obstacles'] for r in rows):.1f} obstacles, "
          f"{_mean(r['moving'] for r in rows):.1f} moving on average)")
    for label in ("full", "no_redirect"):
        reasons = {k: sum(1 for r in rows if r[label] == k) for k in ("ok", "body", "arm")}
        print(f"  {label:12s} solved {reasons['ok']:3d}  failed: no body path {reasons['body']:3d}, "
              f"body path but no arm sequence {reasons['arm']:3d}")
    print(f"  solved only with redirect: {len(rescued)}; solved only without: "
          f"{sum(1 for r in rows if r['full'] != 'ok' and r['no_redirect'] == 'ok')}")
    if both:
        print(f"  path length where both solve: full {_mean(r['full_len'] for r in both):.2f}m, "
              f"no-redirect {_mean(r['no_redirect_len'] for r in both):.2f}m")
    print(f"\n% LaTeX: full {len(full_ok)}/{len(rows)}, no-redirect {len(base_ok)}/{len(rows)}, rescued {len(rescued)}")


def exp_density(n_seeds):
    counts = [0, 2, 4, 6, 8]
    jobs = [(s, p, "roam", {"N_OBSTACLES_RANGE": (n, n)}) for n in counts for p in ("dwa", "oracle")
            for s in range(n_seeds)]
    rows = _pool_map(_run_quiet, jobs)
    _save("density", rows)
    print(f"\n{'drawn':>5s} {'planner':7s} {'actual':>6s} {'done':>6s} {'coll/run':>8s} {'overhead':>8s} "
          f"{'time':>6s} {'redirect':>8s}")
    for n in counts:
        for p in ("dwa", "oracle"):
            sel = [r for r in rows if r["planner"] == p and r["overrides"]["N_OBSTACLES_RANGE"][0] == n]
            s = summarize_runs(sel)
            redirect = s["binding_obstacle"] if p == "dwa" else s["redirect_forced"]
            print(f"{n:5d} {p:7s} {s['obstacles']:6.1f} {s['done']:3d}/{s['n']:<2d} "
                  f"{s['collisions'] / max(s['n'], 1):8.2f} {s['overhead_pct']:7.1f}% {s['time_s']:6.1f} {redirect:8.0%}")
    print("\n% LaTeX rows: drawn & actual & D*Lite+DWA done & coll/run & overhead & oracle done & overhead")
    for n in counts:
        d = summarize_runs([r for r in rows if r["planner"] == "dwa" and r["overrides"]["N_OBSTACLES_RANGE"][0] == n])
        o = summarize_runs([r for r in rows if r["planner"] == "oracle" and r["overrides"]["N_OBSTACLES_RANGE"][0] == n])
        print(f"{n} & {d['obstacles']:.1f} & {d['done']}/{d['n']} & {d['collisions'] / d['n']:.2f} & "
              f"{d['overhead_pct']:.1f}\\% & {o['done']}/{o['n']} & {o['overhead_pct']:.1f}\\% \\\\")


def exp_robustness(n_seeds):
    from config import ROAM_TURN_RATE
    configs = [("no turns", "bounce", 0.0), ("random turns", "bounce", ROAM_TURN_RATE),
               ("no turns", "constant", 0.0), ("random turns", "constant", ROAM_TURN_RATE)]
    # Every override set explicitly in every job: pool workers are reused.
    jobs = [(s, "dwa", "roam", {"ROAM_TURN_RATE": rate, "live_planning.PREDICTION": pred})
            for _, pred, rate in configs for s in range(n_seeds)]
    rows = _pool_map(_run_quiet, jobs)
    _save("robustness", rows)
    print(f"\n{'obstacles':13s} {'prediction':10s} done   coll  runs-w/coll  out-of-bounds  dribble-lost  time(s)")
    for label, pred, rate in configs:
        sel = [r for r in rows if r["overrides"]["ROAM_TURN_RATE"] == rate and r["overrides"]["live_planning.PREDICTION"] == pred]
        s = summarize_runs(sel)
        oob = sum(1 for r in sel if r.get("outcome") == "out_of_bounds")
        lost = sum(1 for r in sel if r.get("outcome") == "dribble_lost")
        print(f"{label:13s} {pred:10s} {s['done']:2d}/{s['n']:<2d} {s['collisions']:5d} {s['runs_with_collision']:8d}"
              f"   {oob:10d}   {lost:10d}     {s['time_s']:6.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["planners", "redirect", "density", "robustness"])
    parser.add_argument("--seeds", type=int, default=None, help="seeds per configuration")
    parser.add_argument("--courses", type=int, default=60, help="raw courses for the redirect ablation")
    args = parser.parse_args()
    if args.experiment == "planners":
        exp_planners(args.seeds or 30)
    elif args.experiment == "redirect":
        exp_redirect(args.courses)
    elif args.experiment == "robustness":
        exp_robustness(args.seeds or 60)
    else:
        exp_density(args.seeds or 20)


if __name__ == "__main__":
    main()
