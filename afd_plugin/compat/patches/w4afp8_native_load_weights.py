# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Apply the GLM-5.2 W4AFP8 routed-expert checkpoint remap in the *native*
``DeepseekV2ForCausalLM.load_weights`` path.

Background
----------
On vLLM 0.19.1 the remap lives in the AFD model wrapper
(``afd_plugin.model_executor.models.deepseek_v2``), which is the active model
class only when AFD is configured. On vLLM >= 0.2x the plugin registers its
wrappers under ``AFD<Arch>`` names and does **not** override the built-in
``GlmMoeDsaForCausalLM`` (a bare ``DeepseekV2ForCausalLM`` subclass), so a plain
(single-instance / non-AFD) GLM-5.2-W4AFP8 run goes through vLLM's native
``DeepseekV2ForCausalLM.load_weights`` and the routed-expert keys are never
remapped -> ``KeyError: ...routed_experts.w2_weight`` (the ``.weight`` suffix
maps to the nonexistent ``w2_weight`` instead of the registered
``w2_weight_packed``).

Fix
---
Monkeypatch ``DeepseekV2ForCausalLM.load_weights`` to apply
``remap_w4afp8_moe_checkpoint_weights`` when the active quantization is
``w4afp8``, then delegate to the original. ``GlmMoeDsaForCausalLM`` inherits
``load_weights`` so patching the base class covers it. The remap is idempotent
(after the first pass keys end in ``.weight_packed`` and are int32, so a second
pass is a no-op), so this coexists safely with the AFD wrapper's own remap on
0.19.1.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _apply() -> None:
    try:
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.models import deepseek_v2
    except Exception:  # vLLM not importable / layout changed
        logger.debug("w4afp8 native load_weights patch: vLLM import failed", exc_info=True)
        return

    target_cls = getattr(deepseek_v2, "DeepseekV2ForCausalLM", None)
    if target_cls is None:
        logger.debug("w4afp8 native load_weights patch: DeepseekV2ForCausalLM not found")
        return

    original = target_cls.load_weights
    if getattr(original, "_afd_w4afp8_wrapped", False):
        return

    from afd_plugin.quantization.w4afp8 import remap_w4afp8_moe_checkpoint_weights

    def load_weights(self, weights):
        vllm_config = get_current_vllm_config()
        quant_config = getattr(vllm_config, "quant_config", None)
        if quant_config is not None and quant_config.get_name() == "w4afp8":
            weights = remap_w4afp8_moe_checkpoint_weights(weights)
        return original(self, weights)

    load_weights._afd_w4afp8_wrapped = True
    target_cls.load_weights = load_weights
    logger.debug("w4afp8 native load_weights patch applied to DeepseekV2ForCausalLM")


_apply()
