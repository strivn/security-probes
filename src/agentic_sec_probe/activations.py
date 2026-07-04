"""Activation extraction from reviewer models via nnsight.

Loads a HuggingFace model (with optional quantization) and extracts
residual stream activations (resid_pre) at all layers with configurable
token pooling.

Uses nnsight instead of TransformerLens because:
- Works with any HuggingFace model (no official model list restriction)
- Supports bitsandbytes quantization
- Qwen2.5-Coder-7B is not in TransformerLens's supported models

Hook pattern resolves to the residual stream before the attention layernorm
(resid_pre). Default: model.model.layers[i].input_layernorm.input[0]
Works for Qwen2, Mistral, Llama2, GPT-BigCode, DeepSeek architectures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction."""

    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    quantization: str | None = None
    token_strategies: tuple[str, ...] = ("mean",)
    device: str = "cuda"
    max_length: int | None = None
    batch_size: int = 8
    n_layers: int = 0
    d_model: int = 0
    hook_pattern: str = "model.layers[{i}].input_layernorm.input[0]"
    dtype: str = "float16"  # full-precision load dtype (Gemma needs "bfloat16")
    hf_load_class: str | None = None  # explicit HF class name (e.g. "Gemma3ForCausalLM")


def load_model(config: ExtractionConfig) -> LanguageModel:
    """Load a model via nnsight, optionally quantized with bitsandbytes.

    quantization=None  → full precision in config.dtype (fp16 default; Gemma="bfloat16")
    quantization="8bit" → bitsandbytes int8, device_map="auto"
    quantization="4bit" → bitsandbytes NF4, device_map="auto"

    config.hf_load_class (e.g. "Gemma3ForCausalLM") names an explicit HF model class so a
    multimodal checkpoint's TEXT-ONLY stack is loaded (vision tower skipped, standard
    model.model.layers[i] hook path preserved). It is loaded via from_pretrained and the
    instance is WRAPPED in nnsight -- NOT passed as nnsight's `automodel=` (concrete classes
    like Gemma3ForCausalLM lack `.from_config`, which nnsight's automodel path requires).
    """
    # Full-precision dtype is config-driven: Gemma overflows in fp16, so it MUST load bf16.
    load_dtype = getattr(torch, config.dtype)
    kwargs: dict[str, object] = {"dtype": load_dtype}

    if config.quantization == "8bit":
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_8bit=True,
        )
    elif config.quantization == "4bit":
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_dtype,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["device_map"] = "auto" if torch.cuda.device_count() > 1 else config.device

    if config.hf_load_class:
        # Load the concrete HF class FIRST, then wrap the instance in nnsight (the automodel=
        # path calls .from_config, which Gemma3ForCausalLM does not implement).
        import transformers
        from transformers import AutoTokenizer

        loader = getattr(transformers, config.hf_load_class)
        hf_model = loader.from_pretrained(config.model_name, **kwargs)
        tok = AutoTokenizer.from_pretrained(config.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = LanguageModel(hf_model, tokenizer=tok)
    else:
        model = LanguageModel(config.model_name, **kwargs)

    # Multimodal configs (Gemma3Config) nest the text params under .text_config; flat configs
    # (Llama/Qwen) expose them directly. Read from whichever has num_hidden_layers.
    hf_config = model._model.config
    if not hasattr(hf_config, "num_hidden_layers") and hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    config.n_layers = hf_config.num_hidden_layers
    config.d_model = hf_config.hidden_size

    return model


# token_strategy -> tensor
# Single sample: [n_layers, d_model]
# Batch: [batch, n_layers, d_model]
PooledActivations = dict[str, torch.Tensor]


def _resolve_hook_proxy(model: LanguageModel, pattern: str, layer_idx: int) -> object:
    """Resolve a hook pattern string to an nnsight proxy during tracing.

    Walks dot-separated path, handling bracket indexing (e.g. layers[3], input[0]).
    Must be called inside a model.trace() context.
    """
    path = pattern.format(i=layer_idx)
    obj: object = model
    for part in path.split("."):
        if "[" in part:
            attr_name, bracket = part.split("[", 1)
            idx = int(bracket.rstrip("]"))
            obj = getattr(obj, attr_name)[idx]
        else:
            obj = getattr(obj, part)
    return obj


@torch.no_grad()
def extract_activations_batch(
    model: LanguageModel,
    codes: list[str],
    config: ExtractionConfig,
) -> PooledActivations:
    """Extract resid_pre activations from multiple code samples.

    Uses nnsight's invoke() to process each sample in a single trace context.
    Each sample is its own forward pass (no padding needed), so pooling
    operates on real tokens only.

    Returns:
        token_strategy -> tensor[batch, n_layers, d_model]
    """
    per_sample_saved: list[list[object]] = []

    with model.trace() as tracer:
        for code in codes:
            with tracer.invoke(code):
                saved: list[object] = []
                for i in range(config.n_layers):
                    proxy = _resolve_hook_proxy(model, config.hook_pattern, i)
                    # Move to CPU INSIDE the trace: nnsight executes proxy ops during
                    # interleaving, so the GPU copy is freed as each layer is produced. This
                    # shrinks the live GPU set from (batch x n_layers) full resid tensors to
                    # ~one layer's worth -- critical for the deep/quantized models (the
                    # batch-wide GPU retention caused 60GB accumulation + OOM on qwen-14b).
                    saved.append(proxy.cpu().save())  # type: ignore[attr-defined]
                per_sample_saved.append(saved)

    # Pool each sample individually: no padding, no attention mask needed. Saved tensors are
    # already on CPU.
    result: dict[str, list[torch.Tensor]] = {s: [] for s in config.token_strategies}

    for saved in per_sample_saved:
        # [n_layers, seq, d_model]
        stacked = torch.stack(list(saved))  # type: ignore[arg-type]
        for s in config.token_strategies:
            result[s].append(_pool_tokens(stacked, s))

    # Stack per-sample results: list of [n_layers, d_model] -> [batch, n_layers, d_model]
    return {s: torch.stack(tensors) for s, tensors in result.items()}


SWIM_WINDOW_SIZES = (16, 32, 64)


def _pool_tokens(
    activations: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    """Reduce [n_layers, seq_len, d_model] to [n_layers, d_model].

    Strategies:
        mean:     mean over all token positions (McKenzie 2506.10805)
        last:     final token position (MoC 2507.09508)
        first:    first token (BOS/CLS)
        max:      max over token positions
        swim_W:   sliding window mean then max (CC++ 2601.04603), W = window size
    """
    if activations.ndim != 3:
        msg = f"Expected [n_layers, seq, d_model] (3D), got shape {activations.shape}"
        raise ValueError(msg)

    if strategy == "mean":
        return activations.mean(dim=1)
    elif strategy == "last":
        return activations[:, -1, :]
    elif strategy == "first":
        return activations[:, 0, :]
    elif strategy == "max":
        return activations.max(dim=1).values
    elif strategy.startswith("swim_"):
        w = int(strategy.split("_")[1])
        seq_len = activations.shape[1]
        w = min(w, seq_len)  # clamp to seq_len (degenerates to max)
        # avg_pool1d computes sliding window means without materializing
        # the full unfolded tensor — O(1) memory vs O(seq * w) for unfold
        # Transpose: [n_layers, seq, d_model] → [n_layers, d_model, seq]
        x = activations.transpose(1, 2)
        windowed_means = torch.nn.functional.avg_pool1d(
            x,
            kernel_size=w,
            stride=1,
        )  # [n_layers, d_model, n_windows]
        return windowed_means.max(dim=2).values  # [n_layers, d_model]
    else:
        msg = f"Unknown pooling strategy: {strategy}"
        raise ValueError(msg)


@torch.no_grad()
def extract_activations(
    model: LanguageModel,
    code: str,
    config: ExtractionConfig,
) -> PooledActivations:
    """Extract activations from a single code sample. Convenience wrapper.

    Returns:
        token_strategy -> tensor[n_layers, d_model]
    """
    batch_result = extract_activations_batch(model, [code], config)
    return {s: t[0] for s, t in batch_result.items()}
