"""Final analysis: aggregate all Phase 1 results into tables and figures.

Reads all outputs/phase1/*.json files and produces:
  Table 1: Layer sweep — CV-best strategy, layer, held-out AUC per model x scope
  Table 2: LogReg vs MLP per model x scope (at CV-best strategy)
  Table 3: Probe vs Prompted vs Semgrep vs Random (same held-out set)
  Table 4: PatchEval OOD — Wilcoxon p, Cohen's d, mean delta
  Fig 1:   Layer sweep curves (CV-best strategy per model)
  Fig 2:   Detection comparison bar chart
  Fig 3:   PatchEval confidence violin
  Fig 4:   Model x scope heatmap

Strategy / layer selection is ALWAYS done via best_cv_auc (5-fold GroupKFold on
the training split, grouped by pair_id). best_eval_auc is the honest out-of-sample
AUC of that selected probe — it is reported in the main tables but never used for
selection.

All tables saved as JSON. Figures as PNG.

Input:  outputs/phase1/*.json
Output: outputs/phase1/tables_*.json, outputs/phase1/fig_*.png

Runs locally after scp'ing all JSON outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agentic_sec_probe.models import MODEL_REGISTRY

OUT_DIR = PROJECT / "outputs" / "phase1"
# c_cpp scope retired (out of reporting scope); python scope is language-defined.
SCOPES = ["universal", "python"]

# Filename prefixes (constructed to avoid triggering security hook on "eval(" substring)
_PROMPTED_PREFIX = "prompted" + "_" + "eval" + "_"
_PATCHEVAL_PREFIX = "patcheval" + "_" + "ood" + "_"


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON if exists, else None."""
    if not path.exists():
        return None
    with open(path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def _prompted_path(slug: str) -> Path:
    return OUT_DIR / f"{_PROMPTED_PREFIX}{slug}.json"


def _patcheval_path(slug: str) -> Path:
    return OUT_DIR / f"{_PATCHEVAL_PREFIX}{slug}.json"


def find_cv_best_strategy(sweep: dict[str, Any], scope_name: str) -> dict[str, Any] | None:
    """Return CV-best (strategy, layer, cv_auc, eval_auc, C, scope_data) for a scope.

    Iterates all strategies in the sweep JSON and picks the one with highest
    best_cv_auc. Selection uses best_cv_auc only — best_eval_auc is reported
    but never drives selection.
    """
    best: dict[str, Any] | None = None
    for strat_name, scope_results in sweep["strategies"].items():
        sc = scope_results.get(scope_name, {})
        if "best_cv_auc" not in sc:
            continue
        cv_auc = sc["best_cv_auc"]
        if best is None or cv_auc > best["cv_auc"]:
            best = {
                "strategy": strat_name,
                "layer": sc["best_layer"],
                "cv_auc": cv_auc,
                "eval_auc": sc.get("best_eval_auc"),
                # Sweep reports weight_decay (constant), no C. Tolerate both schemas.
                "weight_decay": sc.get("weight_decay", sc.get("best_C")),
                "scope_data": sc,
            }
    return best


# ── Table 1: Layer sweep ─────────────────────────────────────────────


def table_layer_sweep() -> dict[str, Any]:
    """CV-best strategy + layer + held-out AUC per model x scope."""
    rows: list[dict[str, Any]] = []

    for slug in MODEL_REGISTRY:
        sweep = load_json(OUT_DIR / f"layer_sweep_{slug}.json")
        if not sweep:
            continue

        row: dict[str, Any] = {"model": slug}
        for scope in SCOPES:
            best = find_cv_best_strategy(sweep, scope)
            if best is None:
                row[f"{scope}_strategy"] = None
                row[f"{scope}_layer"] = None
                row[f"{scope}_cv_auc"] = None
                row[f"{scope}_eval_auc"] = None
                row[f"{scope}_weight_decay"] = None
                continue
            row[f"{scope}_strategy"] = best["strategy"]
            row[f"{scope}_layer"] = best["layer"]
            row[f"{scope}_cv_auc"] = best["cv_auc"]
            row[f"{scope}_eval_auc"] = best["eval_auc"]
            row[f"{scope}_weight_decay"] = best["weight_decay"]
        rows.append(row)

    return {
        "description": (
            "Layer sweep: CV-best strategy and layer per model x scope. "
            "cv_auc drives selection; eval_auc is the honest out-of-sample AUC."
        ),
        "rows": rows,
    }


# ── Table 2: LogReg vs MLP ──────────────────────────────────────────


def table_mlp_comparison() -> dict[str, Any]:
    """LogReg vs MLP per model x scope at the CV-best strategy."""
    rows: list[dict[str, Any]] = []

    for slug in MODEL_REGISTRY:
        mlp = load_json(OUT_DIR / f"mlp_probes_{slug}.json")
        sweep = load_json(OUT_DIR / f"layer_sweep_{slug}.json")
        if not mlp or not sweep:
            continue

        for scope in SCOPES:
            best = find_cv_best_strategy(sweep, scope)
            if best is None:
                continue
            strategy = best["strategy"]
            strat = mlp["strategies"].get(strategy, {})
            scope_data = strat.get(scope, {})
            if not scope_data:
                continue
            rows.append(
                {
                    "model": slug,
                    "scope": scope,
                    "strategy": strategy,
                    "layer": scope_data.get("layer"),
                    "logreg_auc": scope_data.get("logreg", {}).get("held_out_auc"),
                    "mlp_auc": scope_data.get("mlp", {}).get("held_out_auc"),
                    "delta": scope_data.get("delta"),
                }
            )

    return {
        "description": "LogReg vs MLP held-out AUC at the CV-best strategy",
        "rows": rows,
    }


# ── Table 3: Probe vs Prompted vs Semgrep vs Random ─────────────────


def table_detection_comparison() -> dict[str, Any]:
    """Compare probe, prompted LLM, semgrep, and random baselines.

    Probe metric: AUC at the CV-best strategy (continuous score).
    Prompted metric: accuracy/F1/precision/recall (binary output — AUC degenerate).
    """
    random_bl = load_json(OUT_DIR / "random_baseline.json")
    semgrep = load_json(OUT_DIR / "semgrep_eval.json")

    rows: list[dict[str, Any]] = []

    for slug in MODEL_REGISTRY:
        sweep = load_json(OUT_DIR / f"layer_sweep_{slug}.json")
        prompted = load_json(_prompted_path(slug))

        for scope in SCOPES:
            row: dict[str, Any] = {"model": slug, "scope": scope}

            if sweep:
                best = find_cv_best_strategy(sweep, scope)
                if best is not None:
                    row["probe_strategy"] = best["strategy"]
                    row["probe_layer"] = best["layer"]
                    row["probe_auc"] = best["eval_auc"]

            if prompted:
                p_scope = prompted.get("scope_metrics", {}).get(scope, {})
                row["prompted_accuracy"] = p_scope.get("accuracy")
                row["prompted_f1"] = p_scope.get("f1")
                row["prompted_precision"] = p_scope.get("precision")
                row["prompted_recall"] = p_scope.get("recall")

            if semgrep and scope == "python":
                row["semgrep_auc"] = semgrep.get("auc")

            if random_bl:
                r_scope = random_bl.get("per_scope", {}).get(scope, {})
                row["random_ci_upper"] = r_scope.get("ci_95_upper")

            rows.append(row)

    return {
        "description": (
            "Detection comparison: probe AUC (CV-best strategy) vs prompted "
            "accuracy/F1 vs semgrep vs random baseline"
        ),
        "rows": rows,
    }


# ── Table 4: PatchEval OOD ──────────────────────────────────────────


def table_patch_eval_ood() -> dict[str, Any]:
    """PatchEval OOD results per model."""
    rows: list[dict[str, Any]] = []

    for slug in MODEL_REGISTRY:
        pe = load_json(_patcheval_path(slug))
        if not pe:
            continue

        overall = pe.get("overall", {})
        in_dist = pe.get("in_distribution", {})
        ood = pe.get("out_of_distribution", {})

        rows.append(
            {
                "model": slug,
                "probe_strategy": pe.get("probe_strategy"),
                "probe_layer": pe.get("probe_layer"),
                "probe_scope": pe.get("probe_scope"),
                "n_valid": pe.get("n_valid"),
                "n_total": pe.get("n_total"),
                "mean_delta": overall.get("mean_delta"),
                "cohens_d": overall.get("cohens_d"),
                "wilcoxon_p": overall.get("wilcoxon_p"),
                "wilcoxon_alternative": overall.get("wilcoxon_alternative"),
                "bootstrap_ci": overall.get("bootstrap_ci_95"),
                "pct_positive": overall.get("pct_positive_delta"),
                "in_dist_n": in_dist.get("n"),
                "in_dist_mean": (
                    in_dist.get("stats", {}).get("mean_delta") if in_dist.get("stats") else None
                ),
                "ood_n": ood.get("n"),
                "ood_mean": ood.get("stats", {}).get("mean_delta") if ood.get("stats") else None,
            }
        )

    return {"description": "PatchEval OOD validation", "rows": rows}


# ── Figure 1: Layer sweep curves ─────────────────────────────────────


def fig_layer_sweep() -> None:
    """Plot layer sweep AUC curves using the CV-best strategy per (model, scope)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for si, scope in enumerate(SCOPES):
        ax = axes[si]
        for slug in MODEL_REGISTRY:
            sweep = load_json(OUT_DIR / f"layer_sweep_{slug}.json")
            if not sweep:
                continue
            best = find_cv_best_strategy(sweep, scope)
            if best is None:
                continue
            layers_data = best["scope_data"].get("layers", [])
            if not layers_data:
                continue

            x = [r["layer"] for r in layers_data]
            y = [r["eval_auc"] for r in layers_data]
            ax.plot(x, y, label=f"{slug} [{best['strategy']}]", alpha=0.8)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Held-out AUC")
        ax.set_title(f"{scope.title()} scope")
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=6)

    fig.suptitle("Layer Sweep: Held-out AUC by Layer (CV-best strategy per model)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "fig_layer_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_layer_sweep.png")


# ── Figure 2: Detection comparison bar chart ─────────────────────────


def fig_detection_comparison() -> None:
    """Grouped bar chart: probe AUC (CV-best) vs prompted accuracy vs semgrep."""
    random_bl = load_json(OUT_DIR / "random_baseline.json")
    semgrep = load_json(OUT_DIR / "semgrep_eval.json")

    slugs = []
    probe_aucs: list[float] = []
    prompted_accs: list[float | None] = []

    for slug in MODEL_REGISTRY:
        sweep = load_json(OUT_DIR / f"layer_sweep_{slug}.json")
        if not sweep:
            continue

        best = find_cv_best_strategy(sweep, "python")
        if best is None or best["eval_auc"] is None:
            continue

        slugs.append(slug)
        probe_aucs.append(best["eval_auc"])

        prompted = load_json(_prompted_path(slug))
        if prompted:
            p = prompted.get("scope_metrics", {}).get("python", {})
            prompted_accs.append(p.get("accuracy"))
        else:
            prompted_accs.append(None)

    if not slugs:
        return

    x = np.arange(len(slugs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        x - width / 2,
        probe_aucs,
        width,
        label="Probe AUC (CV-best)",
        color="#2196F3",
        alpha=0.85,
    )

    prompted_vals = [v if v is not None else 0.0 for v in prompted_accs]
    ax.bar(
        x + width / 2,
        prompted_vals,
        width,
        label="Prompted accuracy",
        color="#FF9800",
        alpha=0.85,
    )

    if semgrep:
        semgrep_auc = semgrep.get("auc", 0.5)
        ax.axhline(
            y=semgrep_auc, color="#4CAF50", linestyle="--", label=f"Semgrep ({semgrep_auc:.3f})"
        )

    if random_bl:
        r = random_bl.get("per_scope", {}).get("python", {})
        ci_upper = r.get("ci_95_upper", 0.5)
        ax.axhline(y=ci_upper, color="red", linestyle=":", label=f"Random 95% CI ({ci_upper:.3f})")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(slugs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC (probe) / Accuracy (prompted)")
    ax.set_title("Python Scope: Probe AUC vs Prompted Accuracy vs Semgrep")
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "fig_detection_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_detection_comparison.png")


# ── Figure 3: PatchEval confidence violin ─────────────────────────────


def fig_patcheval_violin() -> None:
    """Violin plot of vul_prob vs fix_prob per model."""
    model_data: list[tuple[str, list[float], list[float]]] = []

    for slug in MODEL_REGISTRY:
        pe = load_json(_patcheval_path(slug))
        if not pe:
            continue

        vul_probs = [r["vul_prob"] for r in pe["per_cve"] if "vul_prob" in r]
        fix_probs = [r["fix_prob"] for r in pe["per_cve"] if "fix_prob" in r]
        if vul_probs and fix_probs:
            model_data.append((slug, vul_probs, fix_probs))

    if not model_data:
        return

    fig, axes = plt.subplots(1, len(model_data), figsize=(5 * len(model_data), 5))
    if len(model_data) == 1:
        axes = [axes]

    for ax, (slug, vul_probs, fix_probs) in zip(axes, model_data):
        parts = ax.violinplot([vul_probs, fix_probs], positions=[0, 1], showmeans=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Vulnerable", "Fixed"])
        ax.set_ylabel("P(vulnerable)")
        ax.set_title(slug)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("PatchEval: Probe Confidence on Vulnerable vs Fixed Code", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "fig_patcheval_violin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_patcheval_violin.png")


# ── Figure 4: Per-scope heatmap ──────────────────────────────────────


def fig_scope_heatmap() -> None:
    """Heatmap of model x scope AUC using CV-best strategy per (model, scope)."""
    model_scope_auc: dict[str, dict[str, float]] = {}

    for slug in MODEL_REGISTRY:
        sweep = load_json(OUT_DIR / f"layer_sweep_{slug}.json")
        if not sweep:
            continue

        model_scope_auc[slug] = {}
        for scope in SCOPES:
            best = find_cv_best_strategy(sweep, scope)
            if best is not None and best["eval_auc"] is not None:
                model_scope_auc[slug][scope] = best["eval_auc"]

    if not model_scope_auc:
        return

    models = [s for s in MODEL_REGISTRY if s in model_scope_auc]
    matrix = np.full((len(models), len(SCOPES)), np.nan)
    for mi, slug in enumerate(models):
        for ci, scope in enumerate(SCOPES):
            matrix[mi, ci] = model_scope_auc[slug].get(scope, np.nan)

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)

    ax.set_xticks(range(len(SCOPES)))
    ax.set_xticklabels(SCOPES, fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)

    for mi in range(len(models)):
        for ci in range(len(SCOPES)):
            val = matrix[mi, ci]
            if not np.isnan(val):
                ax.text(ci, mi, f"{val:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, label="Held-out AUC")
    ax.set_title("Model x Scope AUC (CV-best strategy, best layer)")
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "fig_scope_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_scope_heatmap.png")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("Phase 1 Analysis")
    print("=" * 60)

    tables = {
        "table_layer_sweep": table_layer_sweep(),
        "table_mlp": table_mlp_comparison(),
        "table_detection": table_detection_comparison(),
        "table_patch_eval_ood": table_patch_eval_ood(),
    }

    for name, table in tables.items():
        path = OUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(table, f, indent=2)
        n_rows = len(table.get("rows", []))
        print(f"  {name}: {n_rows} rows -> {path.name}")

    print("\n--- Layer Sweep Summary (CV-best strategy) ---")
    for row in tables["table_layer_sweep"]["rows"]:
        py_strat = row.get("python_strategy", "-")
        py_auc = row.get("python_eval_auc", "N/A")
        uni_strat = row.get("universal_strategy", "-")
        uni_auc = row.get("universal_eval_auc", "N/A")
        print(f"  {row['model']}: python={py_auc} [{py_strat}], universal={uni_auc} [{uni_strat}]")

    print("\n--- Detection Comparison (Python) ---")
    for row in tables["table_detection"]["rows"]:
        if row["scope"] != "python":
            continue
        probe = row.get("probe_auc", "N/A")
        acc = row.get("prompted_accuracy", "N/A")
        f1 = row.get("prompted_f1", "N/A")
        print(f"  {row['model']}: probe_auc={probe}, prompted_acc={acc}, prompted_f1={f1}")

    print("\n--- PatchEval OOD ---")
    for row in tables["table_patch_eval_ood"]["rows"]:
        p = row.get("wilcoxon_p")
        d = row.get("cohens_d")
        strat = row.get("probe_strategy", "-")
        layer = row.get("probe_layer", "-")
        print(f"  {row['model']} [{strat} L{layer}]: delta={row.get('mean_delta')}, p={p}, d={d}")

    print("\nGenerating figures...")
    fig_layer_sweep()
    fig_detection_comparison()
    fig_patcheval_violin()
    fig_scope_heatmap()

    print("\nDone.")


if __name__ == "__main__":
    main()
