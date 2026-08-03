#!/usr/bin/env python3
"""Benchmark DeepSeek-V2-Lite AFD sync/async with semantic hard gates."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

from experiment.scripts.run_dsv2_v025_prompt_smoke import (
    DEFAULT_AFD_PORT,
    DEFAULT_API_PORT_BASE,
    DEFAULT_ATTENTION_DEVICES,
    DEFAULT_FFN_DEVICES,
    DEFAULT_MODEL_PATH,
    DEFAULT_VLLM_BIN,
    EXPECTED_VLLM_VERSION,
    MAX_TOKENS,
    PROMPT_CASES,
    comparison_projection,
    parse_device_list,
    validate_response,
)
from tests.e2e.conftest import AFDServer, _launch_afd_server

DEFAULT_OUTPUT_DIR: Final[str] = "/afd-v25/dsv2-v025-async-benchmark"
DEFAULT_NUM_PROMPTS: Final[int] = 64
DEFAULT_NUM_WARMUPS: Final[int] = 8
DEFAULT_MAX_CONCURRENCY: Final[int] = 16
DEFAULT_INPUT_LENGTH: Final[int] = 128
DEFAULT_OUTPUT_LENGTH: Final[int] = 64
DEFAULT_REPETITIONS: Final[int] = 3
DEFAULT_DBO_DECODE_TOKEN_THRESHOLD: Final[int] = 1
DEFAULT_DBO_PREFILL_TOKEN_THRESHOLD: Final[int] = 1
BENCHMARK_TIMEOUT_SECONDS: Final[int] = 1800
SUMMARY_METRICS: Final[tuple[str, ...]] = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vllm-bin", default=DEFAULT_VLLM_BIN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attention-devices", default=DEFAULT_ATTENTION_DEVICES)
    parser.add_argument("--ffn-devices", default=DEFAULT_FFN_DEVICES)
    parser.add_argument("--api-port-base", type=int, default=DEFAULT_API_PORT_BASE)
    parser.add_argument("--afd-port", type=int, default=DEFAULT_AFD_PORT)
    parser.add_argument("--async-transfer", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument("--num-warmups", type=int, default=DEFAULT_NUM_WARMUPS)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
    )
    parser.add_argument("--input-len", type=int, default=DEFAULT_INPUT_LENGTH)
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LENGTH)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--dbo-decode-token-threshold",
        type=int,
        default=DEFAULT_DBO_DECODE_TOKEN_THRESHOLD,
    )
    parser.add_argument(
        "--dbo-prefill-token-threshold",
        type=int,
        default=DEFAULT_DBO_PREFILL_TOKEN_THRESHOLD,
    )
    return parser.parse_args()


def _run_semantic_gate(server: AFDServer) -> list[dict[str, Any]]:
    return [
        validate_response(
            case,
            server.request_completion(
                case.prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            ),
        )
        for case in PROMPT_CASES
    ]


def _validate_reference(
    semantic_results: list[dict[str, Any]],
    reference_path: str | None,
) -> bool | None:
    if reference_path is None:
        return None
    reference_results = json.loads(
        Path(reference_path).read_text(encoding="utf-8"),
    )["results"]
    return comparison_projection(semantic_results) == comparison_projection(
        reference_results,
    )


def _run_benchmark(
    args: argparse.Namespace,
    server: AFDServer,
    output_dir: Path,
    repetition: int,
) -> dict[str, Any]:
    result_filename = f"benchmark-repetition-{repetition}.json"
    command = [
        args.vllm_bin,
        "bench",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(server.attention_port),
        "--model",
        server.served_model,
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "random",
        "--tokenizer",
        args.model,
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.num_warmups),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(args.max_concurrency),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--save-result",
        "--result-dir",
        str(output_dir),
        "--result-filename",
        result_filename,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=BENCHMARK_TIMEOUT_SECONDS,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    log_path = output_dir / f"benchmark-repetition-{repetition}.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"vllm bench repetition {repetition} failed with "
            f"returncode={completed.returncode}; see {log_path}",
        )
    return json.loads((output_dir / result_filename).read_text(encoding="utf-8"))


def _summarize(benchmark_results: list[dict[str, Any]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric in SUMMARY_METRICS:
        values = [
            float(result[metric])
            for result in benchmark_results
            if result.get(metric) is not None
        ]
        if values:
            summary[metric] = {
                "values": values,
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
    return summary


def main() -> int:
    args = parse_args()
    positive_fields = (
        "num_prompts",
        "num_warmups",
        "max_concurrency",
        "input_len",
        "output_len",
        "repetitions",
        "dbo_decode_token_threshold",
        "dbo_prefill_token_threshold",
    )
    for field in positive_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")

    attention_devices = parse_device_list(args.attention_devices, "attention")
    ffn_devices = parse_device_list(args.ffn_devices, "ffn")
    if set(attention_devices) & set(ffn_devices):
        raise ValueError("attention and FFN devices must be disjoint")
    if len(attention_devices) != 1 or len(ffn_devices) != 1:
        raise ValueError("async benchmark currently requires 1A1F topology")
    installed_vllm_version = version("vllm")
    if installed_vllm_version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"expected vLLM {EXPECTED_VLLM_VERSION}, got {installed_vllm_version}",
        )

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server: AFDServer | None = None
    try:
        server = _launch_afd_server(
            backend="gpu",
            model=args.model,
            vllm_bin=args.vllm_bin,
            attention_devices=attention_devices,
            ffn_devices=ffn_devices,
            api_port_base=args.api_port_base,
            afd_port=args.afd_port,
            enable_dbo=True,
            dbo_decode_token_threshold=args.dbo_decode_token_threshold,
            dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
            afd_connector_extra_config=(
                ['{"async_transfer":true,"async_slots":2}']
                if args.async_transfer
                else None
            ),
            common_vllm_args=["--trust-remote-code"],
        )
        semantic_before = _run_semantic_gate(server)
        reference_match_before = _validate_reference(
            semantic_before,
            args.reference,
        )
        if reference_match_before is False:
            raise AssertionError("pre-benchmark semantic output differs from reference")

        benchmark_results = [
            _run_benchmark(args, server, output_dir, repetition)
            for repetition in range(args.repetitions)
        ]

        semantic_after = _run_semantic_gate(server)
        reference_match_after = _validate_reference(
            semantic_after,
            args.reference,
        )
        if reference_match_after is False:
            raise AssertionError(
                "post-benchmark semantic output differs from reference"
            )
    finally:
        if server is not None:
            server.shutdown()

    summary = {
        "model": args.model,
        "topology": "1A1F",
        "mode": "eager",
        "dbo": True,
        "async_transfer": args.async_transfer,
        "vllm_version": installed_vllm_version,
        "workload": {
            "num_prompts": args.num_prompts,
            "num_warmups": args.num_warmups,
            "max_concurrency": args.max_concurrency,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "repetitions": args.repetitions,
            "dbo_decode_token_threshold": args.dbo_decode_token_threshold,
            "dbo_prefill_token_threshold": args.dbo_prefill_token_threshold,
        },
        "semantic_before": semantic_before,
        "semantic_after": semantic_after,
        "reference": args.reference,
        "reference_match_before": reference_match_before,
        "reference_match_after": reference_match_after,
        "metrics": _summarize(benchmark_results),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
