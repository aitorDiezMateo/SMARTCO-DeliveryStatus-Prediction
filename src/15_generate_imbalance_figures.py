"""
Generate publication-quality figures for imbalance analysis and mitigation.

This script:
1) Plots the class distribution from `data/processed/train_raw.csv`.
2) Loads Optuna trials from ONLY `output/optuna/**/*.db`.
3) Creates comparison visualizations for imbalance mitigation:
   - None vs mitigated (based on sampler choice + whether class weights are enabled)
   - Objective distributions by mitigation group
   - Cumulative best curves
   - Usage counts
   - NEW: Effectiveness of each `sampler` option:
     - globally across all models/studies
     - per model type (XGBoost / CatBoost / Bagging / RandomForest)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns


TARGET_COL = "Delivery Status"


def _as_posix_sqlite_url(path: Path) -> str:
    # Optuna expects `sqlite:////absolute/path/to/db.db`.
    return f"sqlite:///{path.as_posix()}"


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def pretty_sampler(s: str) -> str:
    mapping = {
        "none": "None",
        "random_over": "Random Oversampling",
        "random_under": "Random Undersampling",
        "adasyn": "ADASYN",
        "smotenc": "SMOTE-NC",
        "smotenc_tomek": "SMOTE-NC + Tomek Links",
        "smotenc_enn": "SMOTE-NC + ENN",
        "smote": "SMOTE",
        "smote_tomek": "SMOTE + Tomek Links",
        "smote_enn": "SMOTE + ENN",
    }
    return mapping.get(s, s)


def pretty_weights_label(params: dict[str, Any]) -> str:
    # Different models use different weight parameter names.
    if "class_weight" in params:
        v = str(params.get("class_weight"))
        return "class_weight: balanced" if v == "balanced" else f"class_weight: {v}"
    if "class_weight_method" in params:
        v = str(params.get("class_weight_method"))
        mapping = {
            "none": "weights: None",
            "compute_balanced": "weights: balanced (computed)",
            "auto_balanced": "weights: balanced (auto)",
            "auto_sqrt_balanced": "weights: balanced (sqrt auto)",
        }
        return mapping.get(v, f"class_weight_method: {v}")
    if "estimator_class_weight" in params:
        v = str(params.get("estimator_class_weight"))
        return "class weights: balanced" if v == "balanced" else f"estimator_class_weight: {v}"
    if "rf_class_weight" in params:
        v = str(params.get("rf_class_weight"))
        return "class weights: balanced" if v == "balanced" else f"rf_class_weight: {v}"
    return "weights: (unspecified)"


def infer_model_type_from_study_name(study_name: str) -> str:
    s = study_name.lower()
    if "xgboost" in s:
        return "XGBoost"
    if "catboost" in s:
        return "CatBoost"
    # bagging optuna uses study names: bagging__bagging and bagging__rf
    if "bagging__rf" in s or "__rf" in s:
        return "RandomForest"
    if "bagging__bagging" in s or "__bagging" in s:
        return "Bagging"
    if "bagging" in s:
        return "Bagging"
    return study_name


def infer_mitigation_from_trial_params(params: dict[str, Any]) -> dict[str, str]:
    """
    Returns:
      mitigation_group: {none, sampler_only, weights_only}
      mitigation_detail: for legend/tables
    """
    sampler = str(params.get("sampler", "none"))
    sampler_used = sampler != "none"

    # Key project assumption from training code:
    # whenever sampler != "none", weights are forced to "none" to avoid mixing.
    weights_used = False
    if not sampler_used:
        weight_candidates = []
        for k in ("class_weight", "class_weight_method", "estimator_class_weight", "rf_class_weight"):
            if k in params:
                weight_candidates.append(str(params.get(k)))
        weights_used = any(v != "none" for v in weight_candidates) if weight_candidates else False

    if not sampler_used and not weights_used:
        return {"mitigation_group": "none", "mitigation_detail": "none"}
    if sampler_used and not weights_used:
        return {
            "mitigation_group": "sampler_only",
            "mitigation_detail": f"sampler: {pretty_sampler(sampler)}",
        }
    if (not sampler_used) and weights_used:
        return {
            "mitigation_group": "weights_only",
            "mitigation_detail": pretty_weights_label(params),
        }

    # Should be rare given the training logic, but we keep it for completeness.
    return {
        "mitigation_group": "sampler_and_weights",
        "mitigation_detail": f"sampler: {pretty_sampler(sampler)} + {pretty_weights_label(params)}",
    }


def load_class_distribution(train_raw_path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(train_raw_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {train_raw_path}")
    counts = df[target_col].value_counts(dropna=False)
    out = (
        counts.rename("count")
        .to_frame()
        .assign(share=lambda x: x["count"] / x["count"].sum())
        .reset_index()
        .rename(columns={"index": target_col})
    )
    return out.sort_values("count", ascending=False)


def load_optuna_trials(optuna_roots: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for root in optuna_roots:
        if not root.exists():
            continue

        for db_path in sorted(root.rglob("*.db")):
            storage_url = _as_posix_sqlite_url(db_path)
            db_name = db_path.name
            run_id = db_path.parent.name  # e.g. optuna

            try:
                summaries = optuna.get_all_study_summaries(storage=storage_url)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Could not read Optuna DB: {db_path} ({exc})")
                continue

            for s in summaries:
                study_name = s.study_name
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Could not load study '{study_name}' from {db_path}: {exc}")
                    continue

                for t in study.trials:
                    if t.state != optuna.trial.TrialState.COMPLETE:
                        continue
                    if t.value is None:
                        continue

                    sampler = str(t.params.get("sampler", "none"))
                    mitigation = infer_mitigation_from_trial_params(t.params)
                    records.append(
                        {
                            "run_id": run_id,
                            "db": db_name,
                            "study_name": study_name,
                            "trial_number": int(t.number),
                            "value": float(t.value),
                            "sampler": sampler,
                            "sampler_pretty": pretty_sampler(sampler),
                            "mitigation_group": mitigation["mitigation_group"],
                            "mitigation_detail": mitigation["mitigation_detail"],
                            "model_type": infer_model_type_from_study_name(study_name),
                        }
                    )

    if not records:
        raise RuntimeError("No Optuna trials found. Check that `output/optuna/**/*.db` exists and is readable.")

    return pd.DataFrame.from_records(records)


def savefig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_class_distribution(class_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("viridis", n_colors=len(class_df))

    sns.barplot(x=class_df[TARGET_COL].astype(str), y=class_df["count"], palette=palette, ax=ax)
    for i, row in class_df.iterrows():
        ax.text(i, row["count"], f"{row['share']:.1%}", ha="center", va="bottom", fontsize=11)

    ax.set_title("Training Class Distribution")
    ax.set_xlabel("Delivery Status")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    sns.despine(ax=ax)
    savefig(fig, out_dir / "class_distribution.png")


def plot_mitigation_group(df: pd.DataFrame, out_dir: Path) -> None:
    # Main mitigation comparison for the assignment writeup.
    order = ["none", "sampler_only", "weights_only", "sampler_and_weights"]
    present_order = [g for g in order if g in set(df["mitigation_group"])]

    palette = {
        "none": "#7f7f7f",
        "sampler_only": "#4C78A8",
        "weights_only": "#F58518",
        "sampler_and_weights": "#54A24B",
    }

    counts = df["mitigation_group"].value_counts().to_dict()
    xlabels = [f"{g.replace('_', ' ')}\n(n={counts.get(g, 0)})" for g in present_order]
    group_to_xlabel = dict(zip(present_order, xlabels))

    df2 = df.copy()
    df2["mitigation_group_label"] = df2["mitigation_group"].map(group_to_xlabel)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=df2,
        x="mitigation_group_label",
        y="value",
        order=xlabels,
        palette=[palette[g] for g in present_order],
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df2,
        x="mitigation_group_label",
        y="value",
        order=xlabels,
        color="black",
        size=3,
        alpha=0.25,
        jitter=0.18,
        ax=ax,
    )
    ax.set_title("Imbalance Mitigation: Optuna Objective by Choice")
    ax.set_xlabel("Mitigation Choice")
    ax.set_ylabel("Optuna objective value (CV macro-F1)")
    sns.despine(ax=ax)
    savefig(fig, out_dir / "mitigation_group_performance.png")


def plot_none_vs_mitigated(df: pd.DataFrame, out_dir: Path) -> None:
    df2 = df.copy()
    df2["none_vs_mitigated"] = np.where(df2["mitigation_group"] == "none", "none", "mitigated")
    palette = {"none": "#7f7f7f", "mitigated": "#4C78A8"}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df2,
        x="none_vs_mitigated",
        y="value",
        palette=palette,
        inner=None,
        ax=ax,
        cut=0,
        linewidth=0,
    )
    sns.boxplot(
        data=df2,
        x="none_vs_mitigated",
        y="value",
        palette=palette,
        width=0.35,
        showfliers=False,
        ax=ax,
        boxprops={"alpha": 0.9},
    )
    sns.stripplot(
        data=df2,
        x="none_vs_mitigated",
        y="value",
        color="black",
        size=3,
        alpha=0.22,
        jitter=0.14,
        ax=ax,
    )
    ax.set_title("None vs Mitigated: Performance Distribution (CV macro-F1)")
    ax.set_xlabel("")
    ax.set_ylabel("Optuna objective value (CV macro-F1)")
    sns.despine(ax=ax)
    savefig(fig, out_dir / "none_vs_mitigated_distribution.png")


def plot_cumulative_best(df: pd.DataFrame, out_dir: Path) -> None:
    df2 = df.copy()
    df2["none_vs_mitigated"] = np.where(df2["mitigation_group"] == "none", "none", "mitigated")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    palette = {"none": "#7f7f7f", "mitigated": "#4C78A8"}

    for label in ["none", "mitigated"]:
        sub = df2[df2["none_vs_mitigated"] == label].sort_values("trial_number").reset_index(drop=True)
        if sub.empty:
            continue
        x = np.arange(len(sub))
        y = sub["value"].cummax().to_numpy()
        ax.plot(x, y, color=palette[label], linewidth=2.5, label=f"{label} (n={len(sub)})", alpha=0.95)

    ax.set_title("Cumulative Best (Highest CV macro-F1 So Far)")
    ax.set_xlabel("Trial index (within group)")
    ax.set_ylabel("Best-so-far objective value")
    ax.legend(frameon=True)
    sns.despine(ax=ax)
    savefig(fig, out_dir / "cumulative_best_none_vs_mitigated.png")


def plot_method_usage(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["none", "sampler_only", "weights_only", "sampler_and_weights"]
    order = [o for o in order if o in set(df["mitigation_group"])]
    counts = df["mitigation_group"].value_counts().reindex(order).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", n_colors=len(order))
    ax.bar([o.replace("_", " ") for o in order], counts.values, color=colors)
    ax.set_title("How Often Each Mitigation Option Was Tried (Across Optuna Trials)")
    ax.set_xlabel("Mitigation Group")
    ax.set_ylabel("Number of trials")
    sns.despine(ax=ax)
    savefig(fig, out_dir / "mitigation_usage_counts.png")


def plot_top_sampler_details(df: pd.DataFrame, out_dir: Path, *, min_count: int = 10, top_k: int = 10) -> None:
    # Detail-level view based on how the trial achieved mitigation.
    sub = df[df["mitigation_group"].isin(["sampler_only", "weights_only"])].copy()
    if sub.empty:
        return

    cat_counts = sub["mitigation_detail"].value_counts()
    keep = cat_counts[cat_counts >= min_count].index
    sub = sub[sub["mitigation_detail"].isin(keep)]
    if sub.empty:
        return

    summary = (
        sub.groupby("mitigation_detail")["value"]
        .agg(count="count", mean="mean", best="max")
        .sort_values(["best", "mean"], ascending=False)
        .head(top_k)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(summary["mitigation_detail"].astype(str), summary["mean"], color=sns.color_palette("rocket", n_colors=len(summary)))
    ax.set_title(f"Top Mitigation Options (mean CV macro-F1), min_count={min_count}")
    ax.set_xlabel("Mean CV macro-F1")
    ax.set_ylabel("")

    # Mark best.
    for i, row in summary.iterrows():
        ax.scatter(row["best"], i, color="#111111", s=28, zorder=3, marker="D")

    sns.despine(ax=ax)
    savefig(fig, out_dir / "top_mitigation_details.png")


def plot_heatmap_by_model_type(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["mitigation_group"].isin(["none", "sampler_only", "weights_only", "sampler_and_weights"])].copy()
    if sub.empty:
        return

    pivot = (
        sub.groupby(["model_type", "mitigation_group"])["value"]
        .mean()
        .reset_index()
        .pivot(index="model_type", columns="mitigation_group", values="value")
    )

    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", cbar_kws={"label": "Mean CV macro-F1"}, ax=ax)
    ax.set_title("Mean Objective Value by Model Type and Mitigation Group")
    ax.set_xlabel("")
    ax.set_ylabel("Model / Study")
    savefig(fig, out_dir / "heatmap_model_mitigation.png")


def _add_sampler_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the sampler comparison plots, we need to split trials where `sampler="none"` into:
      - `None`: sampler="none" and weights disabled
      - `Class Weights`: sampler="none" and weights_only
    For trials using any resampler, the category is the sampler name.
    """
    df2 = df.copy()
    # Default: use sampler as the category.
    df2["sampler_category"] = df2["sampler"].astype(str)

    mask_none_no_weights = (df2["sampler"] == "none") & (df2["mitigation_group"] == "none")
    df2.loc[mask_none_no_weights, "sampler_category"] = "none"

    mask_class_weights = (df2["sampler"] == "none") & (df2["mitigation_group"] == "weights_only")
    df2.loc[mask_class_weights, "sampler_category"] = "class_weights"

    # Group SMOTE with SMOTENC (including hybrid variants) for visualization clarity.
    # In this project, CatBoost uses labels {"smote", "smote_tomek", "smote_enn"} but internally
    # the implementation is SMOTENC-based (SMOTETomek/SMOTEENN built on SMOTENC).
    canon_map = {
        "smotenc": "smote",
        "smotenc_tomek": "smote_tomek",
        "smotenc_enn": "smote_enn",
    }
    df2["sampler_category"] = df2["sampler_category"].map(lambda s: canon_map.get(s, s))

    return df2


def plot_sampler_global_performance(df: pd.DataFrame, out_dir: Path, *, min_count: int = 10) -> None:
    df2 = _add_sampler_category_column(df)

    counts = df2["sampler_category"].value_counts()
    keep = set(counts[counts >= min_count].index.tolist())
    # Always show assignment-critical categories.
    keep.update({"none", "class_weights"})
    # `ADASYN` can be rare; force it to appear if it exists.
    if "adasyn" in counts.index:
        keep.add("adasyn")

    sub = df2[df2["sampler_category"].isin(keep)].copy()
    if sub.empty:
        return

    order = (
        sub.groupby("sampler_category")["value"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sub["sampler_order"] = sub["sampler_category"].astype(pd.CategoricalDtype(categories=order, ordered=True))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=sub,
        x="sampler_order",
        y="value",
        order=order,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="sampler_order",
        y="value",
        order=order,
        color="black",
        size=3,
        alpha=0.18,
        jitter=0.18,
        ax=ax,
    )

    def pretty_category(cat: str) -> str:
        if cat == "none":
            return "None"
        if cat == "class_weights":
            return "Class Weights"
        return pretty_sampler(cat)

    xlabels = [pretty_category(s) for s in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_title("Effectiveness of Each Sampler (Global Optuna Trials)")
    ax.set_xlabel("Sampler")
    ax.set_ylabel("Optuna objective value (CV macro-F1)")
    sns.despine(ax=ax)
    savefig(fig, out_dir / "sampler_global_performance.png")


def plot_sampler_performance_by_model(df: pd.DataFrame, out_dir: Path, *, min_count: int = 10) -> None:
    df2 = _add_sampler_category_column(df)
    model_types = sorted(set(df["model_type"]))
    if not model_types:
        return

    counts = df2["sampler_category"].value_counts()
    keep = set(counts[counts >= min_count].index.tolist())
    keep.update({"none", "class_weights"})
    # `ADASYN` can be rare; force it to appear if it exists.
    if "adasyn" in counts.index:
        keep.add("adasyn")
    df2 = df2[df2["sampler_category"].isin(keep)].copy()
    if df2.empty:
        return

    # Use global sampler order for consistent visual comparison.
    global_order = df2.groupby("sampler_category")["value"].mean().sort_values(ascending=False).index.tolist()
    n = len(model_types)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, model in enumerate(model_types):
        ax = axes_flat[idx]
        sub = df2[df2["model_type"] == model].copy()
        if sub.empty:
            ax.axis("off")
            continue

        order = [s for s in global_order if s in sub["sampler_category"].unique()]
        sub["sampler_order"] = sub["sampler_category"].astype(pd.CategoricalDtype(categories=order, ordered=True))

        sns.boxplot(data=sub, x="sampler_order", y="value", order=order, showfliers=False, ax=ax)
        sns.stripplot(
            data=sub,
            x="sampler_order",
            y="value",
            order=order,
            color="black",
            size=3,
            alpha=0.18,
            jitter=0.18,
            ax=ax,
        )

        def pretty_category(cat: str) -> str:
            if cat == "none":
                return "None"
            if cat == "class_weights":
                return "Class Weights"
            return pretty_sampler(cat)

        xlabels = [pretty_category(s) for s in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xlabels, rotation=25, ha="right")
        ax.set_title(model)
        ax.set_xlabel("Sampler")
        ax.set_ylabel("Optuna objective (CV macro-F1)")
        sns.despine(ax=ax)

    # Hide unused subplots.
    for j in range(len(model_types), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Effectiveness of Each Sampler (Per Model Type)", y=1.02, fontsize=18)
    savefig(fig, out_dir / "sampler_performance_by_model.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root containing `data/processed` and `output/optuna`.",
    )
    parser.add_argument("--target-col", type=str, default=TARGET_COL)
    parser.add_argument("--min-sampler-count", type=int, default=10)
    args = parser.parse_args()

    set_plot_style()

    project_root = Path(args.project_root)
    out_dir = project_root / "output" / "imbalance_figures"

    # 1) Class distribution (imbalance analysis baseline).
    train_raw_path = project_root / "data" / "processed" / "train_raw.csv"
    class_df = load_class_distribution(train_raw_path, args.target_col)
    plot_class_distribution(class_df, out_dir)

    # 2) Optuna trial loading + mitigation grouping.
    df_trials = load_optuna_trials([project_root / "output" / "optuna"])

    plot_mitigation_group(df_trials, out_dir)
    plot_none_vs_mitigated(df_trials, out_dir)
    plot_cumulative_best(df_trials, out_dir)
    plot_method_usage(df_trials, out_dir)
    plot_top_sampler_details(df_trials, out_dir, min_count=args.min_sampler_count)
    plot_heatmap_by_model_type(df_trials, out_dir)

    # 3) NEW: sampler comparison (requested by assignment).
    plot_sampler_global_performance(df_trials, out_dir, min_count=args.min_sampler_count)
    plot_sampler_performance_by_model(df_trials, out_dir, min_count=args.min_sampler_count)

    # Export CSVs for easy quoting in the writeup.
    out_dir.mkdir(parents=True, exist_ok=True)
    mitigation_summary = (
        df_trials.groupby(["mitigation_group", "model_type"])["value"]
        .agg(trials="count", mean="mean", median="median", best="max", std="std")
        .reset_index()
        .sort_values(["mitigation_group", "model_type"])
    )
    mitigation_summary.to_csv(out_dir / "mitigation_summary_by_group_and_model.csv", index=False)

    df_trials_cat = _add_sampler_category_column(df_trials)

    sampler_summary_global = (
        df_trials_cat.groupby("sampler_category")["value"]
        .agg(trials="count", mean="mean", median="median", best="max", std="std")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    sampler_summary_global.to_csv(out_dir / "sampler_summary_global.csv", index=False)

    sampler_summary_by_model = (
        df_trials_cat.groupby(["model_type", "sampler_category"])["value"]
        .agg(trials="count", mean="mean", median="median", best="max", std="std")
        .reset_index()
        .sort_values(["model_type", "mean"], ascending=[True, False])
    )
    sampler_summary_by_model.to_csv(out_dir / "sampler_summary_by_model.csv", index=False)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

