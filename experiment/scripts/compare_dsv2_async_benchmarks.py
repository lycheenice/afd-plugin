#!/usr/bin/env python3
"""Compare paired DeepSeek-V2-Lite sync/async benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

COMPARISON_METRICS: Final[tuple[str, ...]] = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
)
TARGET_THROUGHPUT_GAIN_PERCENT: Final[float] = 10.0
TARGET_TPOT_REDUCTION_PERCENT: Final[float] = 10.0


def _metric_mean(summary: dict[str, object], metric: str) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    entry = metrics.get(metric)
    if not isinstance(entry, dict) or entry.get("mean") is None:
        return None
    return float(entry["mean"])


def compare_pair(sync_path: Path, async_path: Path) -> dict[str, object]:
    sync_summary = json.loads(sync_path.read_text(encoding="utf-8"))
    async_summary = json.loads(async_path.read_text(encoding="utf-8"))
    if sync_summary["workload"] != async_summary["workload"]:
        raise ValueError(
            f"workloads differ: sync={sync_path}, async={async_path}",
        )
    if async_summary.get("reference_match_before") is not True:
        raise ValueError(f"async pre-benchmark semantic gate failed: {async_path}")
    if async_summary.get("reference_match_after") is not True:
        raise ValueError(f"async post-benchmark semantic gate failed: {async_path}")

    comparisons: dict[str, dict[str, float]] = {}
    for metric in COMPARISON_METRICS:
        sync_mean = _metric_mean(sync_summary, metric)
        async_mean = _metric_mean(async_summary, metric)
        if sync_mean is None or async_mean is None:
            continue
        comparisons[metric] = {
            "sync_mean": sync_mean,
            "async_mean": async_mean,
            "async_vs_sync_percent": (async_mean / sync_mean - 1.0) * 100.0,
        }

    throughput_delta = comparisons["output_throughput"]["async_vs_sync_percent"]
    tpot_delta = comparisons["mean_tpot_ms"]["async_vs_sync_percent"]
    return {
        "sync_summary": str(sync_path),
        "async_summary": str(async_path),
        "workload": sync_summary["workload"],
        "semantic_reference_match_before": True,
        "semantic_reference_match_after": True,
        "metrics": comparisons,
        "performance_gate": {
            "target_throughput_gain_percent": TARGET_THROUGHPUT_GAIN_PERCENT,
            "target_tpot_reduction_percent": TARGET_TPOT_REDUCTION_PERCENT,
            "passed": (
                throughput_delta >= TARGET_THROUGHPUT_GAIN_PERCENT
                or tpot_delta <= -TARGET_TPOT_REDUCTION_PERCENT
            ),
        },
    }


def _parse_pair(value: str) -> tuple[str, Path, Path]:
    label, separator, paths = value.partition("=")
    sync_path, path_separator, async_path = paths.partition(",")
    if not separator or not path_separator or not all((label, sync_path, async_path)):
        raise argparse.ArgumentTypeError(
            "expected LABEL=SYNC_SUMMARY,ASYNC_SUMMARY",
        )
    return label, Path(sync_path), Path(async_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, type=_parse_pair)
    args = parser.parse_args()
    result = {
        label: compare_pair(sync_path, async_path)
        for label, sync_path, async_path in args.pair
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
