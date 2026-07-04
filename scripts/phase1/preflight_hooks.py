"""Verify nnsight hook points for all models in the registry.

For each model that fits on the current GPU:
1. Load with nnsight (using ModelSpec config)
2. Trace a short code string through the model
3. Capture resid_pre at all layers via the hook pattern
4. Verify shape is [n_layers, d_model] — matching ModelSpec expectations
5. Save results to outputs/phase1/preflight_hooks.json

If a model's hook fails, the error is recorded (not raised) so remaining
models still get tested. This is the first script to run after
setting up the model registry — it validates that our activation extraction
will work before we spend hours on the full dataset.
"""

import json
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT / ".env")
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.activations import ExtractionConfig, extract_activations, load_model
from agentic_sec_probe.models import MODEL_REGISTRY, check_gpu_vram

TEST_CODE = "def hello():\n    return 'world'"


def main() -> None:
    vram = check_gpu_vram()
    print(f"GPU VRAM: {vram:.1f} GB")
    results: dict[str, dict[str, object]] = {}

    for slug, spec in MODEL_REGISTRY.items():
        if spec.min_vram_gb > vram:
            print(f"SKIP {spec.name}: needs {spec.min_vram_gb} GB, have {vram:.1f} GB")
            results[slug] = {"status": "skipped", "reason": "insufficient_vram"}
            continue

        print(f"\nTesting {spec.name}...")
        try:
            # Build config from ModelSpec — this is how all Phase 1 scripts
            # will create configs, ensuring hook_pattern comes from the registry
            config = ExtractionConfig(
                model_name=spec.probe_model_id,  # instruct: matches the real extract
                quantization=spec.quantization,
                token_strategies=("mean",),
                hook_pattern=spec.hook_pattern,
                dtype=spec.dtype,
                hf_load_class=spec.hf_load_class,
            )
            model = load_model(config)

            # extract_activations returns {strategy: tensor[n_layers, d_model]}
            acts = extract_activations(model, TEST_CODE, config)
            vec = acts["mean"]

            # Verify against ModelSpec expectations
            expected_shape = (spec.n_layers, spec.d_model)
            actual_shape = tuple(vec.shape)

            if actual_shape != expected_shape:
                print(f"  SHAPE MISMATCH: expected {expected_shape}, got {actual_shape}")
                # Still record as ok but flag the mismatch — the model loaded
                # and hook worked, just dimensions differ from what we expected.
                # This means ModelSpec needs updating, not that the hook is wrong.
                results[slug] = {
                    "status": "shape_mismatch",
                    "expected_n_layers": spec.n_layers,
                    "expected_d_model": spec.d_model,
                    "actual_n_layers": actual_shape[0],
                    "actual_d_model": actual_shape[1],
                    "hook": spec.hook_pattern,
                }
            else:
                results[slug] = {
                    "status": "ok",
                    "n_layers": config.n_layers,
                    "d_model": config.d_model,
                    "hook": spec.hook_pattern,
                    "vec_shape": list(actual_shape),
                }
                print(f"  OK: {actual_shape}")

        except Exception as exc:
            results[slug] = {"status": "error", "error": str(exc)}
            print(f"  FAIL: {exc}")

        finally:
            # Free GPU memory before loading next model
            if "model" in dir():
                del model
            torch.cuda.empty_cache()

    out = PROJECT / "outputs" / "phase1" / "preflight_hooks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # Summary
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    skip = sum(1 for r in results.values() if r["status"] == "skipped")
    fail = sum(1 for r in results.values() if r["status"] == "error")
    mismatch = sum(1 for r in results.values() if r["status"] == "shape_mismatch")
    print(f"\nSummary: {ok} ok, {skip} skipped, {mismatch} shape_mismatch, {fail} error")


if __name__ == "__main__":
    main()
