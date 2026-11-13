"""Out-of-time pitcher-level check for the current pitch-quality SP.

Usage: python validate_pitcher_sp.py pitch_sp_holdout.csv --output validation.json
This is a predictive check, not proof of a causal pitcher contribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(path: Path, min_pitches: int = 60) -> dict:
    df = pd.read_csv(path, parse_dates=["day"])
    cutoff = df.day.min() + (df.day.max() - df.day.min()) / 2
    df["period"] = np.where(df.day <= cutoff, "early", "late")
    # Pitcher-favorable realized value. Pitch SP is the observed offensive
    # count/base-out transition, while pitcher_sp is the pitch-location stage.
    df["realized_defense_sp"] = -df.pitch_sp
    agg = (df.groupby(["pitcher_id", "period"])
           .agg(n=("pitch_sp", "size"), quality=("pitcher_sp", "mean"),
                realized=("realized_defense_sp", "mean"))
           .unstack("period"))
    agg.columns = ["_".join(x) for x in agg.columns]
    eligible = agg.dropna().loc[lambda x:
                              (x.n_early >= min_pitches) &
                              (x.n_late >= min_pitches)].copy()
    result = {"cutoff": str(cutoff.date()), "min_pitches_each_period": min_pitches,
              "eligible_pitchers": len(eligible),
              "target": "later mean defensive observed pitch SP (not causal skill)"}
    if len(eligible) < 8:
        result["status"] = "insufficient_pitchers"
        return result
    # Shrink using early-period pitcher count; shared league prior is zero.
    quality = eligible.quality_early * eligible.n_early / (eligible.n_early + 200)
    baseline = eligible.realized_early * eligible.n_early / (eligible.n_early + 200)
    target = eligible.realized_late
    rank = lambda x: float(pd.Series(x).rank().corr(pd.Series(target).rank()))
    q_rank = rank(quality)
    b_rank = rank(baseline)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        ix = rng.integers(0, len(eligible), len(eligible))
        sub = eligible.iloc[ix]
        q = sub.quality_early * sub.n_early / (sub.n_early + 200)
        b = sub.realized_early * sub.n_early / (sub.n_early + 200)
        t = sub.realized_late
        diffs.append(q.rank().corr(t.rank()) - b.rank().corr(t.rank()))
    diffs = np.asarray(diffs, dtype=float)
    result.update(status="measured_exploratory", quality_future_spearman=q_rank,
                  realized_baseline_future_spearman=b_rank,
                  difference_spearman=q_rank - b_rank,
                  difference_bootstrap_95pct=np.nanquantile(diffs, [.025, .975]).tolist(),
                  passed_baseline_gate=bool(np.nanquantile(diffs, .025) > 0))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pitch_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate(args.pitch_csv)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)

