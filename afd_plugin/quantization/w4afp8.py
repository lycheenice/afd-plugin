# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""W4AFP8 quantization support for GLM-5.2.

This module bridges the ``w4afp8`` quantization method used by GLM-5.2-W4AFP8
checkpoints to vLLM 0.19.1's existing kernel infrastructure:

- **Linear / Attention layers** → delegated to vLLM's native ``Fp8Config``
  (the checkpoint stores these as standard FP8 with ``weight_scale_inv``).
- **MoE expert layers** → delegated to vLLM's native
  ``CompressedTensorsW4A8Fp8MoEMethod`` (CUTLASS W4A8-FP8 kernel).

The only gap is that ``quant_method: w4afp8`` is not in vLLM 0.19.1's
``QUANTIZATION_METHODS`` list.  This module registers it via
``register_quantization_config`` and provides a thin ``W4AFP8Config`` that
routes to the right method per layer type.

See ``experiment/W4AFP8_DESIGN.md`` for full design rationale.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.nn import Module

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.attention import Attention
try:
    # vLLM >= 0.2x: the MoE stack is FusedMoE(factory) -> MoERunner -> RoutedExperts.
    # ``get_quant_method`` is invoked with the RoutedExperts layer (see
    # fused_moe/routed_experts.py::_get_quant_method), so that is the class to match.
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _AFD_MOE_LAYER_CLS,
    )
except ImportError:
    # vLLM 0.19.1: ``FusedMoE`` is itself the MoE layer class.
    from vllm.model_executor.layers.fused_moe import FusedMoE as _AFD_MOE_LAYER_CLS


def _patch_convert_bf16_scales_to_fp8() -> None:
    """Fix vLLM 0.19.1 ``convert_bf16_scales_to_fp8`` bug.

    vLLM 0.19.1 calls ``chan_scales.view(orig_shape[:-1], -1)`` which passes
    a ``torch.Size`` object to ``view()``.  PyTorch's ``view()`` only accepts
    a tuple of ints or individual ints, not a ``torch.Size``, so this raises
    ``TypeError`` for any input dimensionality.  The fix is to unpack with
    ``view(*orig_shape[:-1], -1)``.
    """
    from vllm.model_executor.layers.quantization.utils import quant_utils

    original = quant_utils.convert_bf16_scales_to_fp8
    if getattr(original, "_afd_patched", False):
        return

    def convert_bf16_scales_to_fp8_fixed(quant_fp8, scales):
        import torch
        assert scales.is_contiguous()
        assert scales.is_cuda
        orig_shape = scales.shape
        k_groups = orig_shape[-1]
        flat_scales = scales.view(-1, k_groups)
        fp8_scales, chan_scales = quant_fp8(flat_scales)
        fp8_scales = (fp8_scales.float() / 8.0).to(torch.float8_e4m3fn)
        chan_scales = chan_scales * 8.0
        fp8_scales = fp8_scales.view(orig_shape)
        chan_scales = chan_scales.view(*orig_shape[:-1], -1)
        return fp8_scales, chan_scales

    convert_bf16_scales_to_fp8_fixed._afd_patched = True
    quant_utils.convert_bf16_scales_to_fp8 = convert_bf16_scales_to_fp8_fixed

    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe,
    )
    compressed_tensors_moe.convert_bf16_scales_to_fp8 = (
        convert_bf16_scales_to_fp8_fixed
    )


_patch_convert_bf16_scales_to_fp8()


@register_quantization_config("w4afp8")
class W4AFP8Config(QuantizationConfig):
    """Bridge config for GLM-5.2 W4AFP8 checkpoints.

    Linear/Attention layers use standard FP8 (weight_scale_inv with
    float8_e4m3fn weights, block size [128, 128]).
    MoE expert layers use W4A8-FP8 (int4 weights packed as int8/int32,
    bfloat16 group scales, group_size=128).

    The checkpoint declares ``quant_method: w4afp8`` which vLLM 0.19.1
    does not recognise.  This class registers the method name and routes
    each layer to the appropriate existing vLLM quantization method.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "W4AFP8Config":
        return cls()

    def get_name(self) -> str:
        return "w4afp8"

    def get_config_filenames(self) -> list[str]:
        return []

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def get_quant_method(
        self,
        layer: Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return self._fp8_config.get_quant_method(layer, prefix)
        if isinstance(layer, Attention):
            return self._fp8_config.get_quant_method(layer, prefix)
        if isinstance(layer, _AFD_MOE_LAYER_CLS):
            return _W4AFP8MoEMethod.create(layer.moe_config)
        return None


class _W4AFP8MoEMethod:
    """Factory that creates a ``CompressedTensorsW4A8Fp8MoEMethod``.

    ``CompressedTensorsW4A8Fp8MoEMethod.__init__`` requires
    ``weight_quant`` and ``input_quant`` (compressed-tensors
    ``QuantizationArgs`` objects).  The W4AFP8 checkpoint does not carry a
    compressed-tensors config dict, so we synthesise minimal stand-in
    objects that satisfy the constructor's assertions:
    ``num_bits=4``, ``group_size=128``, ``symmetric=True``.
    """

    @staticmethod
    def create(moe_config) -> Any:
        from compressed_tensors.quantization import QuantizationArgs

        try:
            # vLLM >= 0.2x: split into a per-scheme submodule.
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_fp8 import (  # noqa: E501
                CompressedTensorsW4A8Fp8MoEMethod,
            )
        except ImportError:
            # vLLM 0.19.1: single compressed_tensors_moe module.
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
                CompressedTensorsW4A8Fp8MoEMethod,
            )

        weight_quant = QuantizationArgs(
            num_bits=4,
            group_size=128,
            symmetric=True,
            strategy="group",
            dynamic=False,
            actorder=None,
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            group_size=None,
            symmetric=True,
            strategy="token",
            dynamic=True,
            actorder=None,
        )
        return CompressedTensorsW4A8Fp8MoEMethod(
            weight_quant=weight_quant,
            input_quant=input_quant,
            moe=moe_config,
        )


__all__ = ["W4AFP8Config", "remap_w4afp8_moe_checkpoint_weights"]


# GLM-5.2-W4AFP8 stores routed-expert projections as
# ``...mlp.experts.{id}.{gate,up,down}_proj.weight`` (int8, two uint4b8
# nibbles per byte) with ``...weight_scale_inv`` (bfloat16 group scales).
# vLLM's ``CompressedTensorsW4A8Fp8MoEMethod`` instead registers, during
# loading, ``w13_weight_packed`` / ``w2_weight_packed`` (int32, eight uint4b8
# nibbles per int32) and ``w13_weight_scale`` / ``w2_weight_scale``.  Only the
# routed experts differ; dense MLP, shared experts, attention, and the DSA
# indexer are all FP8 and load through ``Fp8Config`` unchanged.
_ROUTED_EXPERT_PROJ_RE = re.compile(
    r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\."
)

_CKPT_WEIGHT_SUFFIX = ".weight"
_CKPT_SCALE_SUFFIX = ".weight_scale_inv"


def remap_w4afp8_moe_checkpoint_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Rename GLM-5.2 W4AFP8 routed-expert keys to vLLM's W4A8 param names.

    ``make_expert_params_mapping`` builds destination parameter names by string
    substitution that preserves the checkpoint suffix, so genuine
    compressed-tensors checkpoints (``.weight_packed`` / ``.weight_scale``) map
    directly onto ``w13_weight_packed`` / ``w13_weight_scale``.  GLM-5.2 uses
    the plain ``.weight`` / ``.weight_scale_inv`` suffixes, which would map onto
    the nonexistent ``w13_weight`` / ``w13_weight_scale_inv`` and be silently
    skipped, leaving the packed params uninitialised (garbage output).

    This generator rewrites only routed-expert projections:

    - ``.weight`` (int8, 2 nibbles/byte) -> ``.weight_packed`` viewed as int32
      (8 nibbles/int32).  The byte layout is identical, so the int32 view is a
      free reinterpretation; the trailing dimension shrinks by 4x to exactly
      the shape ``CompressedTensorsW4A8Fp8MoEMethod`` allocates.  SGLang stores
      each 4-bit value as signed two's-complement (symmetric ``[-7, 7]``, with
      ``-8`` unused), but vLLM's ``convert_packed_uint4b8_to_signed_int4_inplace``
      subtracts 8 from every nibble, assuming uint4b8 (value + 8) storage.  We
      convert 2's-complement -> uint4b8 by adding 8 modulo 16 to each nibble,
      which is exactly ``XOR 0x8`` per nibble (``0x88`` per byte), so that
      vLLM's later ``- 8`` recovers the correct signed weight.
    - ``.weight_scale_inv`` (bfloat16 group scales) -> ``.weight_scale``.

    Dense MLP, shared experts, attention, and indexer weights are yielded
    unchanged.
    """
    for name, weight in weights:
        if _ROUTED_EXPERT_PROJ_RE.search(name):
            if name.endswith(_CKPT_WEIGHT_SUFFIX) and weight.dtype == torch.int8:
                packed = (weight.contiguous().view(torch.uint8) ^ 0x88).view(
                    torch.int32
                )
                weight = packed
                name = f"{name[: -len(_CKPT_WEIGHT_SUFFIX)]}.weight_packed"
            elif name.endswith(_CKPT_SCALE_SUFFIX):
                name = f"{name[: -len(_CKPT_SCALE_SUFFIX)]}.weight_scale"
        yield name, weight
