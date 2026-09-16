"""Membership inference attack suite against the baseline and DP-trained models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"

SEED = 2026
N_BOOTSTRAP = 1000
FPR_TARGETS = (0.001, 0.01, 0.1)
DELTA = 1e-5
EPS = 1e-12

C_ATTACK = "#eb6834"
C_UTILITY = "#2a78d6"
C_THIRD = "#1baf7a"
C_FOURTH = "#4a3aa7"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

# a label-aware attacker also knows the true outcome. that attacker is stronger.
THREAT_MODELS = {
    "confidence": "label_agnostic",
    "neg_entropy": "label_agnostic",
    "yeom_neg_loss": "label_aware",
    "modified_entropy": "label_aware",
    "class_calibrated_loss": "label_aware",
    "learned_ensemble": "label_aware",
    "lira": "label_aware",
}


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def attack_scores(frame, calibration_reference):
    p = frame["predicted_probability"].to_numpy(dtype=float)
    y = frame["true_label"].to_numpy(dtype=int)
    loss = frame["prediction_loss"].to_numpy(dtype=float)

    p_true = np.where(y == 1, p, 1.0 - p)
    p_true = np.clip(p_true, EPS, 1 - EPS)

    scores = {
        "confidence": np.maximum(p, 1.0 - p),
        "neg_entropy": -binary_entropy(p),
        "yeom_neg_loss": -loss,
        # for two classes this ranks the same as the loss. the auc must agree.
        "modified_entropy": 2.0 * (1.0 - p_true) * np.log(p_true),
    }

    ref_loss = calibration_reference["prediction_loss"].to_numpy(dtype=float)
    ref_y = calibration_reference["true_label"].to_numpy(dtype=int)
    class_median = {c: np.median(ref_loss[ref_y == c]) for c in (0, 1)}
    baseline_loss = np.array([class_median[c] for c in y])
    # remove the difficulty of the class. an easy record must not look like a member.
    scores["class_calibrated_loss"] = -(loss - baseline_loss)

    return scores


def fit_learned_attacker(calibration, reference_for_calibration):
    """Supervised attacker trained on records whose membership it already knows."""
    feats = attack_feature_matrix(calibration, reference_for_calibration)
    model = GradientBoostingClassifier(random_state=SEED, max_depth=2)
    model.fit(feats, calibration["is_member"].to_numpy(dtype=int))
    return model


def attack_feature_matrix(frame, calibration_reference):
    s = attack_scores(frame, calibration_reference)
    return np.column_stack(
        [
            s["confidence"],
            s["neg_entropy"],
            s["yeom_neg_loss"],
            s["class_calibrated_loss"],
            frame["true_label"].to_numpy(dtype=float),
        ]
    )


def tpr_at_fpr(y_true, score, target_fpr):
    """Highest TPR achievable without exceeding target_fpr."""
    fpr, tpr, _ = roc_curve(y_true, score)
    usable = fpr <= target_fpr
    return float(tpr[usable].max()) if usable.any() else 0.0


def bootstrap_ci(y_true, score, statistic, rng, n=N_BOOTSTRAP):
    """Stratified bootstrap so member/non-member counts stay fixed."""
    member_idx = np.flatnonzero(y_true == 1)
    other_idx = np.flatnonzero(y_true == 0)
    draws = []
    for _ in range(n):
        pick = np.concatenate(
            [
                rng.choice(member_idx, member_idx.size, replace=True),
                rng.choice(other_idx, other_idx.size, replace=True),
            ]
        )
        try:
            draws.append(statistic(y_true[pick], score[pick]))
        except ValueError:
            continue
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def balanced_accuracy_at(y_true, score, threshold):
    guess = score >= threshold
    tpr = guess[y_true == 1].mean()
    tnr = (~guess[y_true == 0]).mean()
    return float((tpr + tnr) / 2.0)


def best_threshold(y_true, score):
    """Youden-J threshold, chosen on calibration data only."""
    fpr, tpr, thresholds = roc_curve(y_true, score)
    return float(thresholds[np.argmax(tpr - fpr)])


def clopper_pearson(successes, trials, alpha=0.05):
    """Two-sided exact binomial interval. Keeps the epsilon bound honest."""
    lo = beta.ppf(alpha / 2, successes, trials - successes + 1) if successes > 0 else 0.0
    hi = (
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
        if successes < trials
        else 1.0
    )
    return float(lo), float(hi)


def audit_epsilon(y_true, score, delta=DELTA):
    """Empirical lower bound on epsilon from the attack's confusion matrix."""
    n_member = int((y_true == 1).sum())
    n_other = int((y_true == 0).sum())
    best = 0.0

    for threshold in np.unique(score):
        guess = score >= threshold
        fp = int(guess[y_true == 0].sum())
        fn = int((~guess[y_true == 1]).sum())

        # take the pessimistic end of each interval. the bound must stay valid.
        _, fpr_hi = clopper_pearson(fp, n_other)
        _, fnr_hi = clopper_pearson(fn, n_member)

        if fnr_hi > 0 and (1 - delta - fpr_hi) > 0:
            best = max(best, float(np.log((1 - delta - fpr_hi) / fnr_hi)))

        _, tnr_hi = clopper_pearson(n_other - fp, n_other)
        _, tpr_hi = clopper_pearson(n_member - fn, n_member)
        if tpr_hi > 0 and (1 - delta - tnr_hi) > 0:
            best = max(best, float(np.log((1 - delta - tnr_hi) / tpr_hi)))

    return max(best, 0.0)


def evaluate_target(frame, target_name, rng, lira_lookup=None):
    """Run the full attack suite against one target model's outputs."""
    fold_a, fold_b = train_test_split(
        frame,
        test_size=0.5,
        stratify=frame[["is_member", "true_label"]],
        random_state=SEED,
    )

    scored_frames, scored = [], {}
    # score each half with an attacker calibrated on the other half.
    for calibration, evaluation in ((fold_a, fold_b), (fold_b, fold_a)):
        reference = calibration[calibration["is_member"] == 0]

        fold_scores = attack_scores(evaluation, reference)
        learned = fit_learned_attacker(calibration, reference)
        fold_scores["learned_ensemble"] = learned.predict_proba(
            attack_feature_matrix(evaluation, reference)
        )[:, 1]

        scored_frames.append(evaluation)
        for name, values in fold_scores.items():
            scored.setdefault(name, []).append(values)

    evaluation = pd.concat(scored_frames, ignore_index=True)
    scores = {name: np.concatenate(parts) for name, parts in scored.items()}

    # lira needs shadow models, so it runs in lira.py. measure it with the same code.
    if lira_lookup is not None:
        scores["lira"] = evaluation["record_id"].map(lira_lookup).to_numpy(dtype=float)

    y = evaluation["is_member"].to_numpy(dtype=int)

    rows, curves = [], {}
    for name, score in scores.items():
        auc = roc_auc_score(y, score)
        auc_lo, auc_hi = bootstrap_ci(y, score, roc_auc_score, rng)

        threshold = best_threshold(y, score)

        row = {
            "target": target_name,
            "attack": name,
            "threat_model": THREAT_MODELS[name],
            "auc": float(auc),
            "auc_lo": auc_lo,
            "auc_hi": auc_hi,
            "balanced_accuracy_optimistic": balanced_accuracy_at(y, score, threshold),
            "empirical_epsilon": audit_epsilon(y, score),
            "n_eval": int(len(y)),
            "fpr_resolution_floor": 1.0 / float((y == 0).sum()),
        }
        for target_fpr in FPR_TARGETS:
            row[f"tpr_at_fpr_{target_fpr:g}"] = tpr_at_fpr(y, score, target_fpr)
        lo, hi = bootstrap_ci(y, score, lambda a, b: tpr_at_fpr(a, b, 0.01), rng)
        row["tpr_at_fpr_0.01_lo"] = lo
        row["tpr_at_fpr_0.01_hi"] = hi
        rows.append(row)

        fpr, tpr, _ = roc_curve(y, score)
        curves[name] = (fpr, tpr)

    return rows, curves, evaluation, scores


def subgroup_vulnerability(evaluation, score, patients, target_name, attack_name, rng):
    """Which patients are identifiable? Averages hide the people who get hurt."""
    overlap = [c for c in patients.columns if c != "record_id" and c in evaluation.columns]
    joined = evaluation.drop(columns=overlap).merge(
        patients, on="record_id", how="left", validate="one_to_one"
    )
    joined = joined.assign(score=score)

    non_members = joined[joined["is_member"] == 0]
    members = joined[joined["is_member"] == 1]

    groups = {
        "age": pd.cut(members["age"], [17, 45, 60, 75, 90],
                      labels=["18-45", "46-60", "61-75", "76-90"]),
        "comorbidities": pd.cut(members["comorbidity_count"], [-1, 0, 2, 4, 8],
                                labels=["0", "1-2", "3-4", "5+"]),
        "sex": members["sex"],
        "hospital": members["entity_id"],
        "readmitted": members["true_label"].map({0: "no", 1: "yes"}),
    }

    rows = []
    for dimension, series in groups.items():
        for level, idx in members.groupby(series, observed=True).groups.items():
            subset = members.loc[idx]
            if len(subset) < 30:
                continue
            y = np.concatenate([np.ones(len(subset), int), np.zeros(len(non_members), int)])
            values = np.concatenate([subset["score"].to_numpy(), non_members["score"].to_numpy()])
            auc = float(roc_auc_score(y, values))
            lo, hi = bootstrap_ci(y, values, roc_auc_score, rng, n=400)
            rows.append({
                "target": target_name,
                "attack": attack_name,
                "dimension": dimension,
                "group": str(level),
                "n_members": int(len(subset)),
                "attack_auc": auc,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=13, color=INK, pad=16, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_MUTED)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")
    ax.tick_params(length=0, colors=INK_MUTED, labelsize=9)


def plot_loglog_roc(curves_by_target, path):
    """The chart that shows what average AUC hides."""
    fig, axes = plt.subplots(1, len(curves_by_target), figsize=(5.4 * len(curves_by_target), 5),
                             facecolor=SURFACE, sharey=True)
    colors = [C_ATTACK, C_THIRD, "#eda100", "#e87ba4", C_FOURTH, C_UTILITY]

    # these two curves are the same as two others. drop them before you pair colours.
    redundant = ("modified_entropy", "neg_entropy")

    for ax, (target, curves) in zip(axes, curves_by_target):
        drawn = [(n, c) for n, c in curves.items() if n not in redundant]
        for (name, (fpr, tpr)), color in zip(drawn, colors):
            width = 2.6 if name == "lira" else 2.0
            ax.plot(np.maximum(fpr, 1e-4), np.maximum(tpr, 1e-4), lw=width,
                    color=color, label=name, zorder=3 if name == "lira" else 2)
        ax.plot([1e-4, 1], [1e-4, 1], ls=":", lw=1, color=INK_MUTED, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(1e-3, 1)
        style(ax, target, "False positive rate", "True positive rate")

    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].text(1.2e-3, 1.6e-3, "random guess", fontsize=8, color=INK_MUTED)
    # the dp panel has no lira curve. take the labels from the fullest panel.
    richest = max(axes, key=lambda a: len(a.get_legend_handles_labels()[0]))
    handles, labels = richest.get_legend_handles_labels()
    axes[-1].legend(handles, labels, frameon=False, fontsize=9,
                    loc="lower right", labelcolor=INK_MUTED)
    fig.suptitle(
        "Attack ROC on log-log axes: the low-FPR corner is where patients are identified",
        fontsize=12, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(tradeoff, path, ceiling):
    """Privacy vs utility. Both series are AUCs in [0,1] so they share one axis."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5), facecolor=SURFACE)
    x = np.arange(len(tradeoff))

    ax.fill_between(x, tradeoff["attack_auc_lo"], tradeoff["attack_auc_hi"],
                    color=C_ATTACK, alpha=0.15, lw=0)
    ax.plot(x, tradeoff["attack_auc"], color=C_ATTACK, lw=2, marker="o", ms=8,
            mec=SURFACE, mew=2, label="Privacy risk — strongest attack AUC (95% CI)")
    ax.plot(x, tradeoff["target_test_auc"], color=C_UTILITY, lw=2, marker="s", ms=8,
            mec=SURFACE, mew=2, label="Utility — target model test AUC")

    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=":", zorder=0)
    ax.text(len(x) - 0.55, 0.508, "attack no better than guessing",
            ha="right", fontsize=8, color=INK_MUTED)

    ax.axhline(ceiling, color=C_THIRD, lw=1.5, ls="--", zorder=0)
    ax.text(-0.35, ceiling + 0.008, f"Bayes-optimal ceiling ({ceiling:.3f})",
            fontsize=9, color=C_THIRD)

    for i, row in tradeoff.reset_index(drop=True).iterrows():
        ax.annotate(f"{row['attack_auc']:.2f}", (i, row["attack_auc"]),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, color=INK)
        ax.annotate(f"{row['target_test_auc']:.2f}", (i, row["target_test_auc"]),
                    textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(tradeoff["label"], fontsize=10, color=INK)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(0.42, 0.86)
    style(ax, "Privacy / utility across the privacy budget",
          "\u2190 weaker privacy guarantee        stronger privacy guarantee \u2192", "AUC")
    ax.legend(frameon=False, fontsize=10, loc="upper right", labelcolor=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_epsilon_audit(audit, path, unprotected_epsilon=None):
    """Empirical epsilon vs the epsilon Opacus accounts for."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)

    ax.plot(audit["theoretical_epsilon"], audit["theoretical_epsilon"], ls=":", lw=1.5,
            color=INK_MUTED, zorder=1, label="if the accounting were tight")

    if unprotected_epsilon:
        ax.axhline(unprotected_epsilon, color=C_FOURTH, lw=1.5, ls="--", zorder=1)
        ax.text(audit["theoretical_epsilon"].min(), unprotected_epsilon + 0.18,
                f"unprotected model is caught at eps >= {unprotected_epsilon:.2f}",
                fontsize=9, color=C_FOURTH)

    ax.plot(audit["theoretical_epsilon"], audit["empirical_epsilon"], color=C_ATTACK,
            lw=2.5, marker="o", ms=10, mec=SURFACE, mew=2, zorder=3,
            label="strongest attack's empirical lower bound")

    for _, row in audit.iterrows():
        ax.annotate(f"{row['empirical_epsilon']:.2f}",
                    (row["theoretical_epsilon"], row["empirical_epsilon"]),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    fontsize=10, color=C_ATTACK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_ylim(-0.4, max(float(audit["theoretical_epsilon"].max()),
                          unprotected_epsilon or 0) * 1.15)
    style(ax, "Privacy audit: no attack exceeds the accounted budget",
          "theoretical epsilon (Opacus PRV accountant)", "epsilon lower bound from attack")
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_subgroups(subgroups, path):
    """Who leaks. Averages hide the patients who are actually exposed."""
    data = subgroups[subgroups["dimension"].isin(["age", "comorbidities", "readmitted"])].copy()
    fig, ax = plt.subplots(figsize=(7.2, 7.6), facecolor=SURFACE)
    labels = data["dimension"] + " = " + data["group"]
    y = np.arange(len(data))
    palette = {"age": C_ATTACK, "comorbidities": C_UTILITY, "readmitted": C_FOURTH}
    colors = [palette[d] for d in data["dimension"]]

    ax.barh(y, data["attack_auc"] - 0.5, left=0.5, color=colors, height=0.62)
    ax.errorbar(data["attack_auc"], y,
                xerr=[data["attack_auc"] - data["ci_lo"], data["ci_hi"] - data["attack_auc"]],
                fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=3)
    ax.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.45, max(0.85, float(data["ci_hi"].max()) + 0.03))
    style(ax, "Which patients are identifiable",
          "attack AUC: that group's members vs all non-members", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def load_lira():
    """Per-target LiRA scores, if src/lira.py has been run. Optional by design:"""
    path = OUT_DIR / "lira_scores.csv"
    if not path.exists():
        print("NOTE: results/lira_scores.csv not found; run src/lira.py to include LiRA.")
        return {}
    frame = pd.read_csv(path)
    return {
        target: group.set_index("record_id")["lira_score"]
        for target, group in frame.groupby("target")
    }


def load_patients():
    frames = [
        pd.read_csv(DATA_DIR / name)
        for name in ("hospital_a.csv", "hospital_b.csv")
    ]
    patients = pd.concat(frames, ignore_index=True)
    return patients[["record_id", "age", "sex", "comorbidity_count", "entity_id"]]


def bayes_ceiling():
    """Irreducible AUC on the test split, from the true generative probabilities."""
    oracle = pd.read_csv(DATA_DIR / "oracle.csv")
    splits = pd.read_csv(DATA_DIR / "splits.csv")
    labels = pd.concat(
        [pd.read_csv(DATA_DIR / n)[["record_id", "readmitted_30d"]]
         for n in ("hospital_a.csv", "hospital_b.csv")],
        ignore_index=True,
    )
    joined = labels.merge(oracle, on="record_id").merge(splits, on="record_id")
    test = joined[joined["split"] == "test"]
    return float(roc_auc_score(test["readmitted_30d"], test["true_readmission_probability"]))


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    baseline = pd.read_csv(ROOT / "baseline_predictions.csv")
    dp = pd.read_csv(ROOT / "dp_predictions.csv")
    patients = load_patients()

    baseline_utility = json.loads((ROOT / "baseline_metrics.json").read_text())
    dp_utility = pd.read_csv(ROOT / "dp_metrics.csv").set_index("target_epsilon")
    ceiling = bayes_ceiling()
    lira = load_lira()

    ARM_LABELS = {
        "overfit": "Non-private\n(overfit)",
        "tuned": "Non-private\n(tuned)",
    }

    all_rows, all_subgroups, curves_by_target = [], [], []
    utility, theoretical, plot_labels = {}, {}, {}

    for arm in ("overfit", "tuned"):
        name = f"Non-private ({arm})"
        rows, curves, evaluation, scores = evaluate_target(
            baseline[baseline["arm"] == arm], name, rng, lira.get(name)
        )
        all_rows += rows
        best = max(rows, key=lambda r: r["auc"])["attack"]
        all_subgroups.append(
            subgroup_vulnerability(evaluation, scores[best], patients, name, best, rng)
        )
        utility[name] = baseline_utility[arm]["test_auc"]
        theoretical[name] = np.inf
        plot_labels[name] = ARM_LABELS[arm]
        curves_by_target.append((name, curves))

    dp_curves_for_plot = None
    for epsilon, group in sorted(dp.groupby("target_epsilon"), reverse=True):
        name = f"DP eps={epsilon:g}"
        rows, curves, evaluation, scores = evaluate_target(group, name, rng, lira.get(name))
        all_rows += rows
        best = max(rows, key=lambda r: r["auc"])["attack"]
        all_subgroups.append(
            subgroup_vulnerability(evaluation, scores[best], patients, name, best, rng)
        )
        utility[name] = float(dp_utility.loc[epsilon, "test_auc"])
        theoretical[name] = float(dp_utility.loc[epsilon, "actual_epsilon"])
        plot_labels[name] = f"DP\neps={epsilon:g}"
        if epsilon == 3.0:
            dp_curves_for_plot = (name, curves)

    results = pd.DataFrame(all_rows)
    subgroups = pd.concat(all_subgroups, ignore_index=True)
    results.to_csv(OUT_DIR / "mia_results.csv", index=False)
    subgroups.to_csv(OUT_DIR / "subgroup_vulnerability.csv", index=False)

    # rank on auc. at 1000 non-members the low-fpr numbers are too noisy to rank on.
    strongest = (
        results.sort_values("auc", ascending=False)
        .groupby("target", as_index=False)
        .first()
    )

    order = ["Non-private (overfit)", "Non-private (tuned)",
             "DP eps=8", "DP eps=3", "DP eps=1", "DP eps=0.5"]
    strongest = strongest.set_index("target").loc[order].reset_index()

    tradeoff = pd.DataFrame([
        {
            "label": plot_labels[row["target"]],
            "target": row["target"],
            "attack": row["attack"],
            "attack_auc": row["auc"],
            "attack_auc_lo": row["auc_lo"],
            "attack_auc_hi": row["auc_hi"],
            "attack_tpr_at_1pct": row["tpr_at_fpr_0.01"],
            "target_test_auc": utility[row["target"]],
            "pct_of_bayes_ceiling": utility[row["target"]] / ceiling,
            "theoretical_epsilon": theoretical[row["target"]],
            "empirical_epsilon": row["empirical_epsilon"],
        }
        for _, row in strongest.iterrows()
    ])
    tradeoff.to_csv(OUT_DIR / "privacy_utility_tradeoff.csv", index=False)

    plot_tradeoff(tradeoff, ROOT / "privacy_utility_tradeoff.png", ceiling)
    plot_loglog_roc(curves_by_target + [dp_curves_for_plot], OUT_DIR / "mia_roc_loglog.png")
    unprotected = tradeoff.loc[
        ~np.isfinite(tradeoff["theoretical_epsilon"]), "empirical_epsilon"
    ].max()
    plot_epsilon_audit(tradeoff[np.isfinite(tradeoff["theoretical_epsilon"])],
                       OUT_DIR / "epsilon_audit.png", float(unprotected))
    plot_subgroups(subgroups[subgroups["target"] == "Non-private (overfit)"],
                   OUT_DIR / "subgroup_vulnerability.png")

    pd.set_option("display.width", 220)
    print("\nATTACK SUITE  (cross-fitted over every record, per target)")
    print("=" * 118)
    view = results[["target", "attack", "threat_model", "auc", "auc_lo", "auc_hi",
                    "tpr_at_fpr_0.01", "empirical_epsilon"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n\nHEADLINE  (Bayes-optimal test AUC on this data = {ceiling:.4f})")
    print("=" * 118)
    print(tradeoff.drop(columns=["label", "attack_tpr_at_1pct"])
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n\nPRIVACY AUDIT")
    print("=" * 118)
    for _, row in tradeoff.iterrows():
        if not np.isfinite(row["theoretical_epsilon"]):
            print(f"  {row['target']:22s}  no formal guarantee; "
                  f"attack certifies eps >= {row['empirical_epsilon']:.2f}")
            continue
        verdict = "OK" if row["empirical_epsilon"] <= row["theoretical_epsilon"] else "VIOLATION"
        print(f"  {row['target']:22s}  accounted eps={row['theoretical_epsilon']:.2f}  "
              f"empirical eps>={row['empirical_epsilon']:.2f}  [{verdict}]")

    overfit_attack = tradeoff.loc[0, "attack"]
    print(f"\n\nMOST EXPOSED PATIENT GROUPS (overfit model, {overfit_attack}, per-group AUC)")
    print("=" * 118)
    top = (subgroups[subgroups["target"] == "Non-private (overfit)"]
           .sort_values("attack_auc", ascending=False).head(10))
    print(top[["dimension", "group", "n_members", "attack_auc", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    floor = float(results["fpr_resolution_floor"].iloc[0])
    print(f"\nNOTE: with {int(1 / floor)} non-members the finest resolvable FPR is "
          f"{floor:.1%}; TPR @ 0.1% FPR sits at that floor and is reported for "
          f"completeness only.")
    print(f"\nWrote {OUT_DIR}/ and privacy_utility_tradeoff.png")


if __name__ == "__main__":
    main()
