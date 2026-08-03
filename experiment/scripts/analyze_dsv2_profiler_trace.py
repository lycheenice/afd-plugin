#!/usr/bin/env python3
"""Summarize compute/communication overlap in PyTorch profiler traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

NCCL_KERNEL_MARKER = "nccl"
TRACE_EVENT_PHASE = "X"
KERNEL_CATEGORY = "kernel"


class Interval(TypedDict):
    start_us: float
    end_us: float


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: item["start_us"]):
        if merged and interval["start_us"] <= merged[-1]["end_us"]:
            merged[-1]["end_us"] = max(merged[-1]["end_us"], interval["end_us"])
        else:
            merged.append(interval.copy())
    return merged


def _interval_duration(intervals: list[Interval]) -> float:
    return sum(item["end_us"] - item["start_us"] for item in intervals)


def _intersection_duration(left: list[Interval], right: list[Interval]) -> float:
    left_index = 0
    right_index = 0
    duration_us = 0.0
    while left_index < len(left) and right_index < len(right):
        duration_us += max(
            0.0,
            min(left[left_index]["end_us"], right[right_index]["end_us"])
            - max(left[left_index]["start_us"], right[right_index]["start_us"]),
        )
        if left[left_index]["end_us"] <= right[right_index]["end_us"]:
            left_index += 1
        else:
            right_index += 1
    return duration_us


def analyze_trace(trace_path: Path) -> dict[str, object]:
    trace = json.loads(trace_path.read_text())
    compute_intervals: list[Interval] = []
    communication_intervals: list[Interval] = []
    stream_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"compute_kernels": 0, "communication_kernels": 0}
    )

    for event in trace["traceEvents"]:
        args = event.get("args", {})
        if (
            event.get("cat") != KERNEL_CATEGORY
            or event.get("ph") != TRACE_EVENT_PHASE
            or event.get("dur") is None
            or "stream" not in args
        ):
            continue

        interval = Interval(
            start_us=float(event["ts"]),
            end_us=float(event["ts"] + event["dur"]),
        )
        stream = str(args["stream"])
        if NCCL_KERNEL_MARKER in str(event.get("name", "")).lower():
            communication_intervals.append(interval)
            stream_counts[stream]["communication_kernels"] += 1
        else:
            compute_intervals.append(interval)
            stream_counts[stream]["compute_kernels"] += 1

    merged_compute = _merge_intervals(compute_intervals)
    merged_communication = _merge_intervals(communication_intervals)
    compute_streams = sorted(
        stream for stream, counts in stream_counts.items() if counts["compute_kernels"]
    )
    communication_streams = sorted(
        stream
        for stream, counts in stream_counts.items()
        if counts["communication_kernels"]
    )

    return {
        "trace": str(trace_path),
        "compute_streams": compute_streams,
        "communication_streams": communication_streams,
        "shared_compute_communication_streams": sorted(
            set(compute_streams) & set(communication_streams)
        ),
        "stream_kernel_counts": dict(sorted(stream_counts.items())),
        "compute_union_us": _interval_duration(merged_compute),
        "communication_union_us": _interval_duration(merged_communication),
        "compute_communication_overlap_us": _intersection_duration(
            merged_compute, merged_communication
        ),
    }


def _parse_trace_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=TRACE_PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        type=_parse_trace_argument,
        help="labeled profiler trace in LABEL=TRACE_PATH form; repeat as needed",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = {label: analyze_trace(path) for label, path in args.trace}
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
