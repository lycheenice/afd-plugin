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
from vllm.model_executor.layers.fused_moe import FusedMoE


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
        if isinstance(layer, FusedMoE):
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
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
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


__all__ = ["W4AFP8Config"]
