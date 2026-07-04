"""Model registry and GPU VRAM guard.

Single source of truth for model configs used across all Phase 1 scripts.
Each ModelSpec contains everything needed to load a model, extract activations,
and run prompted inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single model in the registry.

    quantization: None for fp16, "8bit" for bitsandbytes int8, "4bit" for NF4.
    """

    name: str
    slug: str
    hf_base_id: str
    hf_instruct_id: str
    n_layers: int
    d_model: int
    quantization: str | None
    min_vram_gb: float
    hook_pattern: str = "model.layers[{i}].input_layernorm.input[0]"
    # Full-precision load dtype when quantization is None. Default fp16; Gemma-2/3 MUST be
    # bf16 (fp16 overflows -> NaN/<pad> garbage, documented). Ignored when quantized.
    dtype: str = "float16"
    # Optional explicit HF model class for from_pretrained (e.g. "Gemma3ForCausalLM" to load
    # the TEXT-ONLY stack of a multimodal checkpoint, skipping the vision tower and keeping
    # the standard model.model.layers[i] hook path). None -> AutoModelForCausalLM.
    hf_load_class: str | None = None

    @property
    def probe_model_id(self) -> str:
        """Checkpoint to extract PROBE activations from.

        Returns the INSTRUCT id so the probe reads the SAME checkpoint that the
        prompted baseline (05/12) is scored on — making "same model" literally
        true. Previously probe scripts read hf_base_id while
        prompting read hf_instruct_id, an undisclosed base/instruct confound for
        Qwen-7B/14B and DeepSeek-33B (Devstral's base==instruct so it was the only
        matched control). Lit note: re-run the layer/pooling/C CV from scratch on
        these instruct activations; do NOT port the base-model layer choice
        (fine-tuning shifts features — arXiv:2202.10054).
        """
        return self.hf_instruct_id


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen2.5-coder-7b": ModelSpec(
        name="Qwen2.5-Coder-7B",
        slug="qwen2.5-coder-7b",
        hf_base_id="Qwen/Qwen2.5-Coder-7B",
        hf_instruct_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        n_layers=28,
        d_model=3584,
        quantization=None,
        min_vram_gb=16.0,
    ),
    "qwen2.5-coder-14b": ModelSpec(
        name="Qwen2.5-Coder-14B",
        slug="qwen2.5-coder-14b",
        hf_base_id="Qwen/Qwen2.5-Coder-14B",
        hf_instruct_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        n_layers=48,
        d_model=5120,
        # fp16 on the 80GB H100 (~28GB): avoids the intermittent bitsandbytes int8 kernel
        # crash (ops.cu:388) that aborted 8-bit extraction, and is higher precision. 8-bit
        # was an A100-40GB constraint.
        quantization=None,
        min_vram_gb=30.0,
    ),
    "devstral-small": ModelSpec(
        name="Devstral-Small-2505",
        slug="devstral-small",
        hf_base_id="mistralai/Devstral-Small-2505",
        hf_instruct_id="mistralai/Devstral-Small-2505",
        n_layers=40,
        d_model=5120,
        quantization=None,  # fp16 (~24GB), fits 80GB; avoids the int8 kernel crash
        min_vram_gb=28.0,
    ),
    "starcoder2-15b": ModelSpec(
        name="StarCoder2-15B",
        slug="starcoder2-15b",
        hf_base_id="bigcode/starcoder2-15b",
        hf_instruct_id="bigcode/starcoder2-15b-instruct-v0.1",
        n_layers=40,
        d_model=6144,
        quantization="8bit",
        min_vram_gb=18.0,
    ),
    "codellama-13b": ModelSpec(
        name="CodeLlama-13B",
        slug="codellama-13b",
        hf_base_id="codellama/CodeLlama-13b-hf",
        hf_instruct_id="codellama/CodeLlama-13b-Instruct-hf",
        n_layers=40,
        d_model=5120,
        quantization="8bit",
        min_vram_gb=15.0,
    ),
    "deepseek-coder-33b": ModelSpec(
        name="DeepSeek-Coder-33B",
        slug="deepseek-coder-33b",
        hf_base_id="deepseek-ai/deepseek-coder-33b-base",
        hf_instruct_id="deepseek-ai/deepseek-coder-33b-instruct",
        n_layers=62,
        d_model=7168,
        # fp16 (~66GB) on the 80GB H100 -- TRY fp16 first (avoids the int8 kernel crash +
        # quant confound); fall back to 8-bit if it OOMs with activations.
        # Python-only extraction (760 samples) keeps the activation working set small.
        quantization=None,
        min_vram_gb=70.0,
    ),
    "qwen2.5-coder-32b": ModelSpec(
        name="Qwen2.5-Coder-32B",
        slug="qwen2.5-coder-32b",
        hf_base_id="Qwen/Qwen2.5-Coder-32B",
        hf_instruct_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        n_layers=64,
        d_model=5120,
        quantization=None,
        min_vram_gb=65.0,
    ),
    "llama-3.3-70b": ModelSpec(
        name="Llama-3.3-70B",
        slug="llama-3.3-70b",
        hf_base_id="meta-llama/Llama-3.3-70B-Instruct",
        hf_instruct_id="meta-llama/Llama-3.3-70B-Instruct",
        n_layers=80,
        d_model=8192,
        quantization=None,
        min_vram_gb=145.0,
    ),
    # Cross-family additions: Llama-3 and Gemma-3, both fit a 40GB A100.
    # Standard Llama arch -> standard hook, fp16-safe. Gemma-3 -> bf16 ONLY + text-only
    # load class (Gemma3ForCausalLM) so the SigLIP vision tower is skipped and the hook
    # stays on the standard model.model.layers[i] path.
    "llama-3.1-8b": ModelSpec(
        name="Llama-3.1-8B",
        slug="llama-3.1-8b",
        hf_base_id="meta-llama/Llama-3.1-8B",
        hf_instruct_id="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
        d_model=4096,
        quantization=None,  # fp16 ~16GB, fits 40GB
        min_vram_gb=18.0,
    ),
    "gemma-3-12b": ModelSpec(
        name="Gemma-3-12B",
        slug="gemma-3-12b",
        hf_base_id="google/gemma-3-12b-pt",
        hf_instruct_id="google/gemma-3-12b-it",
        n_layers=48,
        d_model=3840,
        quantization=None,  # bf16 ~25GB (incl. ~0.8GB vision tower we don't hook), fits 80GB
        min_vram_gb=30.0,
        dtype="bfloat16",  # fp16 -> NaN garbage on Gemma; bf16 mandatory
        # The -it checkpoint is MULTIMODAL: text weights live under language_model.* on disk,
        # so the text-only Gemma3ForCausalLM class re-inits them to RANDOM (verified: HF logs
        # "newly initialized"). Must load the full ConditionalGeneration class and hook the
        # NESTED text layers. Path verified empirically by preflight_hooks (transformers 5.5).
        hf_load_class="Gemma3ForConditionalGeneration",
        hook_pattern="model.language_model.layers[{i}].input_layernorm.input[0]",
    ),
}


def check_gpu_vram() -> float:
    """Return total GPU VRAM in GB (summed across all devices)."""
    if not torch.cuda.is_available():
        return 0.0
    total = sum(
        torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())
    )
    return float(total / (1024**3))


def _apply_model_filter() -> None:
    """Filter MODEL_REGISTRY by ASP_MODEL_FILTER env var at import time.

    ASP_MODEL_FILTER: comma-separated slugs, e.g. "qwen2.5-coder-32b,llama-3.3-70b".
    If unset, registry is unchanged. All scripts that import MODEL_REGISTRY
    automatically see only the filtered models.
    """
    filt = os.environ.get("ASP_MODEL_FILTER", "")
    if not filt:
        return
    slugs = {s.strip() for s in filt.split(",") if s.strip()}
    for key in list(MODEL_REGISTRY.keys()):
        if key not in slugs:
            del MODEL_REGISTRY[key]
    print(f"ASP_MODEL_FILTER active: {sorted(MODEL_REGISTRY.keys())}")


_apply_model_filter()


def get_available_models(max_vram_gb: float | None = None) -> list[ModelSpec]:
    """Return models that fit within the given VRAM budget, ordered by size."""
    specs = sorted(MODEL_REGISTRY.values(), key=lambda s: s.min_vram_gb)
    if max_vram_gb is not None:
        specs = [s for s in specs if s.min_vram_gb <= max_vram_gb]
    return specs
