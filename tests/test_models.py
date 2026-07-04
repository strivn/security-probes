"""Tests for model registry and GPU VRAM guard."""

from agentic_sec_probe.models import MODEL_REGISTRY, get_available_models


def test_registry_has_all_models() -> None:
    assert len(MODEL_REGISTRY) >= 5
    assert "qwen2.5-coder-7b" in MODEL_REGISTRY
    assert "qwen2.5-coder-14b" in MODEL_REGISTRY
    assert "devstral-small" in MODEL_REGISTRY
    assert "starcoder2-15b" in MODEL_REGISTRY
    assert "codellama-13b" in MODEL_REGISTRY
    assert "deepseek-coder-33b" in MODEL_REGISTRY


def test_model_spec_fields() -> None:
    spec = MODEL_REGISTRY["qwen2.5-coder-7b"]
    assert spec.n_layers == 28
    assert spec.d_model == 3584
    assert spec.quantization is None
    assert spec.min_vram_gb > 0
    assert isinstance(spec.hook_pattern, str)
    assert "{i}" in spec.hook_pattern


def test_model_spec_is_frozen() -> None:
    spec = MODEL_REGISTRY["qwen2.5-coder-7b"]
    try:
        spec.name = "modified"  # type: ignore[misc]
        raise AssertionError("Should not allow mutation")
    except AttributeError:
        pass  # expected — frozen dataclass


def test_get_available_models_filters_by_vram() -> None:
    small = get_available_models(max_vram_gb=10.0)
    all_models = get_available_models(max_vram_gb=100.0)
    assert len(small) < len(all_models)
    for m in small:
        assert m.min_vram_gb <= 10.0


def test_get_available_models_no_filter() -> None:
    all_models = get_available_models()
    assert len(all_models) == len(MODEL_REGISTRY)


def test_get_available_models_ordered_by_vram() -> None:
    models = get_available_models(max_vram_gb=100.0)
    vrams = [m.min_vram_gb for m in models]
    assert vrams == sorted(vrams)


def test_slug_is_filesystem_safe() -> None:
    for slug in MODEL_REGISTRY:
        assert "/" not in slug
        assert " " not in slug
        assert slug == slug.lower()


def test_slug_matches_key() -> None:
    for slug, spec in MODEL_REGISTRY.items():
        assert slug == spec.slug


def test_probe_model_id_is_instruct() -> None:
    """Probe activations must come from the INSTRUCT checkpoint so probe and
    prompted baseline use the same model."""
    for spec in MODEL_REGISTRY.values():
        assert spec.probe_model_id == spec.hf_instruct_id


def test_base_instruct_differ_for_known_confounded_models() -> None:
    """Document the confound: these models had base != instruct, so the old
    probe (base) and prompted (instruct) were different checkpoints."""
    for slug in ("qwen2.5-coder-7b", "qwen2.5-coder-14b", "deepseek-coder-33b"):
        spec = MODEL_REGISTRY[slug]
        assert spec.hf_base_id != spec.hf_instruct_id
    # Devstral is the matched control: base == instruct.
    devstral = MODEL_REGISTRY["devstral-small"]
    assert devstral.hf_base_id == devstral.hf_instruct_id
