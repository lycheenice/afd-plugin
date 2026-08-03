#!/usr/bin/env python3
"""Run a small deterministic DeepSeek-V2-Lite AFD prompt matrix on GPU."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

from tests.e2e.conftest import AFDServer, _launch_afd_server

DEFAULT_MODEL_PATH: Final[str] = "/models/DeepSeek-V2-Lite"
DEFAULT_VLLM_BIN: Final[str] = "/usr/local/bin/vllm"
DEFAULT_OUTPUT_PATH: Final[str] = "/afd-v25/dsv2-v025-prompt-smoke.json"
EXPECTED_VLLM_VERSION: Final[str] = "0.25.0"
DEFAULT_ATTENTION_DEVICES: Final[str] = "0"
DEFAULT_FFN_DEVICES: Final[str] = "1"
DEFAULT_API_PORT_BASE: Final[int] = 18100
DEFAULT_AFD_PORT: Final[int] = 6339
MAX_TOKENS: Final[int] = 16
DEFAULT_REPETITIONS: Final[int] = 1
DEFAULT_CONCURRENCY: Final[int] = 1


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    required_prefix: str
    required_fragments: tuple[str, ...]


PROMPT_CASES: Final[tuple[PromptCase, ...]] = (
    PromptCase("The capital of France is", "paris", ("paris",)),
    PromptCase("1 + 1 =", "2", ("2",)),
    PromptCase("San Francisco is a", "city", ("city", "neighborhood")),
    PromptCase(
        "Write a Python function that adds two integers:",
        "def ",
        ("def ", "return", "+"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vllm-bin", default=DEFAULT_VLLM_BIN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--attention-devices", default=DEFAULT_ATTENTION_DEVICES)
    parser.add_argument("--ffn-devices", default=DEFAULT_FFN_DEVICES)
    parser.add_argument("--api-port-base", type=int, default=DEFAULT_API_PORT_BASE)
    parser.add_argument("--afd-port", type=int, default=DEFAULT_AFD_PORT)
    parser.add_argument("--enable-dbo", action="store_true")
    parser.add_argument("--async-transfer", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
    )
    return parser.parse_args()


def parse_device_list(raw_devices: str, role: str) -> list[str]:
    devices = [device.strip() for device in raw_devices.split(",") if device.strip()]
    if not devices:
        raise ValueError(f"{role} device list must not be empty")
    if any(not device.isdigit() for device in devices):
        raise ValueError(f"invalid {role} device list: {raw_devices!r}")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate {role} device in: {raw_devices!r}")
    return devices


def validate_response(case: PromptCase, body: dict[str, Any]) -> dict[str, Any]:
    prompt = case.prompt
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"unexpected choices for {prompt!r}: {choices!r}")
    choice = choices[0]
    text = choice.get("text")
    if not isinstance(text, str) or not text.strip():
        raise AssertionError(f"empty completion for {prompt!r}: {text!r}")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise AssertionError(
            f"unexpected finish_reason for {prompt!r}: {finish_reason!r}",
        )

    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise AssertionError(f"missing usage for {prompt!r}")
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", 0))
    if prompt_tokens <= 0 or completion_tokens <= 0:
        raise AssertionError(f"invalid usage for {prompt!r}: {usage!r}")
    if total_tokens != prompt_tokens + completion_tokens:
        raise AssertionError(f"inconsistent usage for {prompt!r}: {usage!r}")

    normalized_text = text.strip().lower()
    if not normalized_text.startswith(case.required_prefix.lower()):
        raise AssertionError(
            f"completion for {prompt!r} does not directly continue the context; "
            f"expected_prefix={case.required_prefix!r}, text={text!r}",
        )
    missing_fragments = [
        fragment
        for fragment in case.required_fragments
        if fragment.lower() not in normalized_text
    ]
    if missing_fragments:
        raise AssertionError(
            f"completion for {prompt!r} is not semantically connected; "
            f"missing={missing_fragments!r}, text={text!r}",
        )

    return {
        "prompt": prompt,
        "text": text,
        "required_prefix": case.required_prefix,
        "required_fragments": list(case.required_fragments),
        "finish_reason": finish_reason,
        "usage": usage,
    }


def comparison_projection(results: object) -> list[dict[str, Any]]:
    if not isinstance(results, list) or not all(
        isinstance(result, dict) for result in results
    ):
        raise ValueError("reference results must be a list of objects")
    return [
        {
            "prompt": result["prompt"],
            "text": result["text"],
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
        }
        for result in results
    ]


def main() -> int:
    args = parse_args()
    if args.async_transfer and not args.enable_dbo:
        raise ValueError("--async-transfer requires --enable-dbo")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    attention_devices = parse_device_list(args.attention_devices, "attention")
    ffn_devices = parse_device_list(args.ffn_devices, "ffn")
    overlapping_devices = sorted(set(attention_devices) & set(ffn_devices))
    if overlapping_devices:
        raise ValueError(
            "attention and FFN devices must be disjoint; "
            f"overlap={overlapping_devices!r}",
        )
    topology = f"{len(attention_devices)}A{len(ffn_devices)}F"
    installed_vllm_version = version("vllm")
    if installed_vllm_version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"expected vLLM {EXPECTED_VLLM_VERSION}, got {installed_vllm_version}",
        )
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    server: AFDServer | None = None
    results: list[dict[str, Any]] = []
    try:
        server = _launch_afd_server(
            backend="gpu",
            model=args.model,
            vllm_bin=args.vllm_bin,
            attention_devices=attention_devices,
            ffn_devices=ffn_devices,
            api_port_base=args.api_port_base,
            afd_port=args.afd_port,
            enable_dbo=args.enable_dbo,
            dbo_decode_token_threshold=1,
            dbo_prefill_token_threshold=1,
            afd_connector_extra_config=(
                ['{"async_transfer":true,"async_slots":2}']
                if args.async_transfer
                else None
            ),
            common_vllm_args=["--trust-remote-code"],
        )
        request_specs = [
            (repetition, case)
            for repetition in range(args.repetitions)
            for case in PROMPT_CASES
        ]

        def request_and_validate(
            repetition: int,
            case: PromptCase,
        ) -> dict[str, Any]:
            body = server.request_completion(
                case.prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            result = validate_response(case, body)
            result["repetition"] = repetition
            return result

        if args.concurrency == 1:
            results = [
                request_and_validate(repetition, case)
                for repetition, case in request_specs
            ]
        else:
            ordered_results: list[dict[str, Any] | None] = [None] * len(request_specs)
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        request_and_validate,
                        repetition,
                        case,
                    ): request_index
                    for request_index, (repetition, case) in enumerate(request_specs)
                }
                for future in as_completed(futures):
                    request_index = futures[future]
                    ordered_results[request_index] = future.result()
            if any(result is None for result in ordered_results):
                raise RuntimeError("concurrent prompt result set is incomplete")
            results = [result for result in ordered_results if result is not None]

        for result in results:
            print(
                f"repetition={result['repetition']} prompt={result['prompt']!r} "
                f"completion={result['text']!r}",
                flush=True,
            )
    finally:
        if server is not None:
            server.shutdown()

    reference_match: bool | None = None
    if args.reference is not None:
        reference_results = json.loads(
            Path(args.reference).read_text(encoding="utf-8"),
        ).get("results")
        expected_results = comparison_projection(reference_results) * args.repetitions
        reference_match = comparison_projection(results) == expected_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "topology": topology,
                "mode": "eager",
                "dbo": args.enable_dbo,
                "async_transfer": args.async_transfer,
                "reference": args.reference,
                "reference_match": reference_match,
                "repetitions": args.repetitions,
                "concurrency": args.concurrency,
                "vllm_version": installed_vllm_version,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if reference_match is False:
        raise AssertionError(
            "prompt completions differ from the synchronous reference; "
            f"observed output is saved at {output_path}",
        )
    print(
        f"PASS: topology={topology} {len(results)} prompt completions; "
        f"result={output_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
