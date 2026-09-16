# build the 5 slide presentation for the gisec final, from the results files.

from pathlib import Path
import json

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "presentation"

TEAM = "unlisted"

# dark surface reads better than white on a large stage screen.
INK = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0xC3, 0xC2, 0xB7)
FAINT = RGBColor(0x8A, 0x89, 0x80)
SURFACE = RGBColor(0x1A, 0x1A, 0x19)
PANEL = RGBColor(0x26, 0x26, 0x24)
BLUE = RGBColor(0x39, 0x87, 0xE5)
ORANGE = RGBColor(0xD9, 0x59, 0x26)
GREEN = RGBColor(0x19, 0x9E, 0x70)
VIOLET = RGBColor(0x90, 0x85, 0xE9)

W = Inches(13.333)
H = Inches(7.5)
MARGIN = Inches(0.72)


# the report charts are drawn on a light surface. a stage screen needs the dark
# version, with labels big enough to read from the back of the hall.
CHART_BG = "#1a1a19"
CHART_INK = "#ffffff"
CHART_MUTED = "#c3c2b7"
CHART_GRID = "#33332f"


def _dark_axes(ax, fig):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.grid(color=CHART_GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#4a4a46")
    ax.tick_params(colors=CHART_MUTED, labelsize=13, length=0)


def chart_tradeoff(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.read_csv(RESULTS / "privacy_utility_tradeoff.csv")
    order = ["Non-private (overfit)", "Non-private (tuned)",
             "DP eps=8", "DP eps=3", "DP eps=1", "DP eps=0.5"]
    t = t.set_index("target").loc[order].reset_index()
    labels = ["No care", "Careful", "DP 8", "DP 3", "DP 1", "DP 0.5"]
    x = range(len(t))

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    _dark_axes(ax, fig)

    ax.errorbar(x, t["target_test_auc_mean"], yerr=t["target_test_auc_std"],
                color="#3987e5", lw=3.2, marker="s", ms=11, mec=CHART_BG, mew=2,
                capsize=5, label="Accuracy of the model")
    ax.errorbar(x, t["attack_auc_mean"], yerr=t["attack_auc_std"],
                color="#d95926", lw=3.2, marker="o", ms=11, mec=CHART_BG, mew=2,
                capsize=5, label="Success of the attack")

    ax.axhline(0.5, color="#6f6e68", lw=1.6, ls=":")
    ax.text(len(t) - 0.45, 0.512, "attacker learns nothing", ha="right",
            fontsize=12, color="#8a8980")

    ax.annotate("", xy=(1, 0.516), xytext=(0, 0.578),
                arrowprops=dict(arrowstyle="-|>", color="#199e70", lw=2.6))
    ax.text(0.52, 0.60, "the leak\ncollapses", fontsize=13, color="#199e70",
            fontweight="bold", ha="center")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=13, color=CHART_INK)
    ax.set_ylim(0.44, 0.83)
    ax.set_ylabel("score", fontsize=13, color=CHART_MUTED)
    ax.legend(frameon=False, fontsize=13, loc="upper right",
              labelcolor=CHART_MUTED, handlelength=1.6)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=200, facecolor=CHART_BG)
    plt.close(fig)


def chart_subgroups(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    g = pd.read_csv(RESULTS / "subgroup_vulnerability.csv")
    g = g[g["target"] == "Non-private (overfit)"]
    g = g[g["dimension"].isin(["comorbidities", "age"])]
    pretty = {"comorbidities": "conditions", "age": "age"}
    g = g.assign(label=[f"{pretty[d]} {v}" for d, v in zip(g["dimension"], g["group"])])
    g = g.sort_values("attack_auc_mean")

    colours = ["#d95926" if v >= 0.65 else "#5b5a55" for v in g["attack_auc_mean"]]
    y = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    _dark_axes(ax, fig)
    ax.barh(y, g["attack_auc_mean"] - 0.5, left=0.5, color=colours, height=0.66)
    ax.axvline(0.5, color="#6f6e68", lw=1.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(g["label"], fontsize=13, color=CHART_INK)
    ax.set_xlim(0.5, 0.78)
    ax.set_xticks([0.5, 0.6, 0.7])
    ax.set_xlabel("how easily the attack finds them", fontsize=13, color=CHART_MUTED)
    ax.grid(axis="y", visible=False)

    for index, value in enumerate(g["attack_auc_mean"]):
        ax.text(value + 0.006, index, f"{value:.2f}", va="center", fontsize=12,
                color=CHART_INK if value >= 0.65 else CHART_MUTED,
                fontweight="bold" if value >= 0.65 else "normal")

    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=200, facecolor=CHART_BG)
    plt.close(fig)


def picture(slide, path, left, top, width, height):
    """drop an image in, keeping its aspect ratio inside the given box."""
    from PIL import Image

    with Image.open(path) as image:
        ratio = image.width / image.height
    if width / height > ratio:
        draw_h, draw_w = height, Emu(int(height * ratio))
    else:
        draw_w, draw_h = width, Emu(int(width / ratio))
    return slide.shapes.add_picture(
        str(path), left + Emu(int((width - draw_w) / 2)),
        top + Emu(int((height - draw_h) / 2)), draw_w, draw_h)


def load():
    t = pd.read_csv(RESULTS / "privacy_utility_tradeoff.csv").set_index("target")
    r = pd.read_csv(RESULTS / "mia_results.csv")
    g = pd.read_csv(RESULTS / "subgroup_vulnerability.csv")
    g = g[g["target"] == "Non-private (overfit)"].sort_values(
        "attack_auc_mean", ascending=False
    )
    seeds = sorted((ROOT / "runs").glob("seed*"))
    base = json.loads((seeds[0] / "baseline_metrics.json").read_text())
    dp = pd.read_csv(seeds[0] / "dp_metrics.csv").set_index("target_epsilon")
    cohort = sum(
        len(pd.read_csv(ROOT / "data" / f"hospital_{h}.csv")) for h in ("a", "b")
    )
    return t, r, g, base, dp, len(seeds), cohort


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def background(slide, colour=SURFACE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = colour


def box(slide, left, top, width, height, fill=PANEL, line=None, line_width=Pt(1)):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = line_width
    shape.shadow.inherit = False
    return shape


def text(slide, left, top, width, height, runs, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.0, space_after=0):
    """runs: list of (string, size_pt, colour, bold) or None for a blank line."""
    frame = slide.shapes.add_textbox(left, top, width, height).text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0

    first = True
    for item in runs:
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.alignment = align
        paragraph.line_spacing = spacing
        if space_after:
            paragraph.space_after = Pt(space_after)
        if item is None:
            paragraph.text = ""
            continue
        body, size, colour, bold = item
        run = paragraph.add_run()
        run.text = body
        run.font.size = Pt(size)
        run.font.color.rgb = colour
        run.font.bold = bold
        run.font.name = "Arial"
    return frame


def eyebrow(slide, label):
    text(slide, MARGIN, Inches(0.42), W - 2 * MARGIN, Inches(0.3),
         [(label, 12, BLUE, True)])


def headline(slide, line, top=Inches(0.86), size=38):
    text(slide, MARGIN, top, W - 2 * MARGIN, Inches(1.0),
         [(line, size, INK, True)], spacing=0.95)


def footer(slide, line):
    text(slide, MARGIN, H - Inches(0.62), W - 2 * MARGIN, Inches(0.3),
         [(line, 12, FAINT, False)])


def stat_card(slide, left, top, width, height, value, colour, caption, accent=True):
    card = box(slide, left, top, width, height)
    if accent:
        bar = box(slide, left, top, Inches(0.05), height, fill=colour)
        bar.line.fill.background()
    pad = Inches(0.28)
    text(slide, left + pad, top + Inches(0.26), width - 2 * pad, Inches(0.8),
         [(value, 40, colour, True)])
    text(slide, left + pad, top + Inches(1.02), width - 2 * pad,
         height - Inches(1.2), [(caption, 14, MUTED, False)], spacing=1.12)


def bullet_card(slide, left, top, width, height, title, lines, accent=BLUE,
                body_size=13.5):
    box(slide, left, top, width, height)
    bar = box(slide, left, top, Inches(0.05), height, fill=accent)
    bar.line.fill.background()
    pad = Inches(0.28)

    text(slide, left + pad, top + Inches(0.22), width - 2 * pad, Inches(0.38),
         [(title, 16, INK, True)])
    text(slide, left + pad, top + Inches(0.72), width - 2 * pad,
         height - Inches(0.92),
         [(line, body_size, MUTED, False) for line in lines],
         spacing=1.08, space_after=7)


# ---------------------------------------------------------------- slides


def slide_title(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    slide = blank(prs)
    background(slide)

    eyebrow(slide, "SCHOOL OF CYBER DEFENSE 2026   ·   GISEC GLOBAL FINAL")

    text(slide, MARGIN, Inches(1.9), W - 2 * MARGIN, Inches(1.8),
         [("The model is the leak.", 66, INK, True)], spacing=0.92)

    text(slide, MARGIN, Inches(3.5), Inches(8.4), Inches(1.1),
         [("Pseudonymisation removes the name from the record. We proved that the "
           "trained model still gives the patient away.", 21, MUTED, False)],
         spacing=1.2)

    text(slide, MARGIN, Inches(4.8), Inches(8.4), Inches(0.5),
         [("Team ", 17, FAINT, False)])
    text(slide, MARGIN + Inches(0.62), Inches(4.8), Inches(6.0), Inches(0.5),
         [(TEAM, 17, INK, True)])

    # the three numbers the jury asked us to improve
    strip_top = Inches(5.8)
    card_w = Inches(3.83)
    gap = Inches(0.25)
    for index, (value, caption, colour) in enumerate([
        (f"{cohort:,}", "patients, three times our Stage 2 cohort", BLUE),
        (f"{n_seeds} runs", "every model trained and attacked three times", VIOLET),
        ("0.6%", "accuracy we pay for a formal privacy guarantee", GREEN),
    ]):
        left = MARGIN + index * (card_w + gap)
        box(slide, left, strip_top, card_w, Inches(1.05))
        text(slide, left + Inches(0.26), strip_top + Inches(0.16), card_w, Inches(0.45),
             [(value, 24, colour, True)])
        text(slide, left + Inches(0.26), strip_top + Inches(0.6), card_w - Inches(0.4),
             Inches(0.4), [(caption, 11.5, MUTED, False)])


def slide_objective(prs, data):
    slide = blank(prs)
    background(slide)
    eyebrow(slide, "THE PROBLEM")
    headline(slide, "The law moves the model, not the data.\nWe assumed that was safe.", size=36)

    top = Inches(2.35)
    height = Inches(3.35)
    left_w = Inches(5.1)
    right_w = W - 2 * MARGIN - left_w - Inches(0.3)

    bullet_card(slide, MARGIN, top, left_w, Inches(1.58),
                "The records cannot travel",
                ["Federal Law No. 2 of 2019. UAE health data must not leave the State.",
                 "So we send the trained model instead."], BLUE)
    bullet_card(slide, MARGIN, top + Inches(1.77), left_w, Inches(1.58),
                "The names are already gone",
                ["PDPL Art. 20 asks for pseudonymisation.",
                 "No name, no address, no file number ever leaves the hospital."], ORANGE)

    right = MARGIN + left_w + Inches(0.3)
    box(slide, right, top, right_w, height)
    bar = box(slide, right, top, Inches(0.05), height, fill=VIOLET)
    bar.line.fill.background()
    pad = Inches(0.34)
    text(slide, right + pad, top + Inches(0.34), right_w - 2 * pad, Inches(0.4),
         [("So we asked one question", 15, VIOLET, True)])
    text(slide, right + pad, top + Inches(0.84), right_w - 2 * pad, Inches(1.5),
         [("Can an outsider tell whether YOUR record was used to train this model?",
           27, INK, True)], spacing=1.06)
    text(slide, right + pad, top + Inches(2.35), right_w - 2 * pad, Inches(0.9),
         [("If the answer is yes, the model has disclosed that you were a patient at "
           "that hospital, in that period. No diagnosis required.", 14, MUTED, False)],
         spacing=1.12)

    box(slide, MARGIN, Inches(6.0), W - 2 * MARGIN, Inches(0.82))
    text(slide, MARGIN + Inches(0.34), Inches(6.19), W - 2 * MARGIN - Inches(0.68),
         Inches(0.55),
         [("Our job was not to claim that this gap exists. It was to measure it.",
           19, INK, True)])


def slide_solution(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    slide = blank(prs)
    background(slide)
    eyebrow(slide, "THE SOLUTION")
    headline(slide, "Records stay home. Only protected maths travels.")

    # architecture strip
    top = Inches(2.05)
    height = Inches(1.35)
    node_w = Inches(3.5)
    arrow_w = Inches(0.62)
    left = MARGIN

    def node(x, title, subtitle, colour):
        box(slide, x, top, node_w, height)
        bar = box(slide, x, top, Inches(0.05), height, fill=colour)
        bar.line.fill.background()
        text(slide, x + Inches(0.26), top + Inches(0.26), node_w - Inches(0.5),
             Inches(0.4), [(title, 16, INK, True)])
        text(slide, x + Inches(0.26), top + Inches(0.7), node_w - Inches(0.5),
             Inches(0.5), [(subtitle, 12.5, MUTED, False)], spacing=1.1)

    node(left, "Hospital A",
         "3,000 patients. Trains locally, with noise at every step.", BLUE)
    text(slide, left + node_w, top + Inches(0.42), arrow_w, Inches(0.5),
         [("⇄", 26, FAINT, False)], align=PP_ALIGN.CENTER)
    node(left + node_w + arrow_w, "Aggregation server",
         "Averages the protected updates. Never sees a record.", ORANGE)
    text(slide, left + 2 * node_w + arrow_w, top + Inches(0.42), arrow_w, Inches(0.5),
         [("⇄", 26, FAINT, False)], align=PP_ALIGN.CENTER)
    node(left + 2 * (node_w + arrow_w), "Hospital B",
         "3,000 patients. Trains locally, with noise at every step.", BLUE)

    text(slide, MARGIN, top + Inches(1.42), W - 2 * MARGIN, Inches(0.3),
         [("Eight rounds. The protected update goes up, the averaged model comes "
           "back. The records never move.", 12, FAINT, False)],
         align=PP_ALIGN.CENTER)

    # three claims
    card_w = Inches(3.83)
    gap = Inches(0.25)
    ctop = Inches(3.75)
    for index, (title, lines, accent) in enumerate([
        ("The server is not trusted",
         ["Noise goes into every training step, inside the hospital.",
          "A stolen server learns no more than the epsilon bound allows."], GREEN),
        ("The budget does not add up",
         ["No patient is at both hospitals.",
          "The cost is the larger of the two, not the sum."], BLUE),
        ("Scale makes privacy cheap",
         ["In Stage 2, at 2,000 patients, privacy cost 2.5% of the accuracy.",
          "At 6,000 patients it costs 0.6%."], VIOLET),
    ]):
        bullet_card(slide, MARGIN + index * (card_w + gap), ctop, card_w,
                    Inches(1.85), title, lines, accent)

    box(slide, MARGIN, Inches(5.85), W - 2 * MARGIN, Inches(0.85))
    text(slide, MARGIN + Inches(0.34), Inches(6.02), W - 2 * MARGIN - Inches(0.68),
         Inches(0.6),
         [("More data does not make privacy more expensive. It makes it cheaper.",
           19, GREEN, True)])


def slide_validation(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    slide = blank(prs)
    background(slide)
    eyebrow(slide, "RED TEAM")
    headline(slide, "We attacked ourselves. Who got hurt?")

    overfit = t.loc["Non-private (overfit)"]
    dp3 = t.loc["DP eps=3"]
    ov = r[r["target"] == "Non-private (overfit)"]
    lira = ov[ov["attack"] == "lira"].iloc[0]
    rival = ov[ov["attack"] != "lira"].sort_values("auc_mean", ascending=False).iloc[0]

    top = Inches(1.95)
    chart_w = Inches(5.45)
    picture(slide, OUT_DIR / "chart_subgroups.png", MARGIN, top, chart_w, Inches(3.9))
    text(slide, MARGIN, top + Inches(3.95), chart_w, Inches(0.3),
         [("Careless model. 0.5 means the attacker is guessing.", 11.5, FAINT, False)],
         align=PP_ALIGN.CENTER)

    right = MARGIN + chart_w + Inches(0.34)
    right_w = W - MARGIN - right

    pair_h = Inches(1.52)
    half = Emu(int((right_w - Inches(0.22)) / 2))
    for index, (value, colour, caption) in enumerate([
        (f'{overfit["attack_auc_mean"]:.3f}', ORANGE,
         "against the careless model. Our audit certifies a real leak at epsilon "
         f'{overfit["empirical_epsilon_mean"]:.2f}.'),
        (f'{dp3["attack_auc_mean"]:.3f}', GREEN,
         f'against the DP model. Chance level in all {n_seeds} runs, spread '
         f'{dp3["attack_auc_std"]:.3f}.'),
    ]):
        x = right + index * (half + Inches(0.22))
        box(slide, x, top, half, pair_h)
        text(slide, x + Inches(0.24), top + Inches(0.18), half - Inches(0.4),
             Inches(0.55), [(value, 34, colour, True)])
        text(slide, x + Inches(0.24), top + Inches(0.74), half - Inches(0.4),
             Inches(0.7), [(caption, 11.5, MUTED, False)], spacing=1.08)

    bullet_card(slide, right, top + pair_h + Inches(0.2), right_w, Inches(1.72),
                "Seven attacks, including the state of the art",
                [f"LiRA builds {32 * n_seeds} shadow models for each target, so it "
                 "knows what a normal score looks like for every single patient.",
                 f"At a 1% false alarm rate it finds "
                 f"{lira['tpr_01_mean'] / rival['tpr_01_mean']:.1f} times more people "
                 "than the classic attack."], VIOLET, body_size=12.5)

    punch_top = top + pair_h + Inches(2.12)
    box(slide, right, punch_top, right_w, Inches(1.14))
    bar = box(slide, right, punch_top, Inches(0.05), Inches(1.14), fill=ORANGE)
    bar.line.fill.background()
    text(slide, right + Inches(0.3), punch_top + Inches(0.22), right_w - Inches(0.6),
         Inches(0.8),
         [("The attack is best at finding the patients with the most to lose.",
           17, INK, True)], spacing=1.1)

    footer(slide, "LiRA: Carlini et al., IEEE Symposium on Security and Privacy, 2022. "
                  f"Mean of {n_seeds} independent runs.")


def slide_results(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    tuned_rows = r[r["target"] == "Non-private (tuned)"]
    tuned_lira = float(tuned_rows[tuned_rows["attack"] == "lira"]["auc_mean"].iloc[0])
    overfit, tuned, dp3 = (t.loc["Non-private (overfit)"], t.loc["Non-private (tuned)"],
                           t.loc["DP eps=3"])
    actual = float(dp.loc[3.0, "actual_epsilon"])

    slide = blank(prs)
    background(slide)
    eyebrow(slide, "RESULTS AND RECOMMENDATION")
    headline(slide, "A formal guarantee for 0.6% of the accuracy.")

    top = Inches(1.95)
    chart_w = Inches(6.35)
    picture(slide, OUT_DIR / "chart_tradeoff.png", MARGIN, top, chart_w, Inches(3.85))
    text(slide, MARGIN, top + Inches(3.9), chart_w, Inches(0.3),
         [(f"Mean of {n_seeds} runs. The best score any model could reach is 0.784.",
           11.5, FAINT, False)], align=PP_ALIGN.CENTER)

    right = MARGIN + chart_w + Inches(0.34)
    right_w = W - MARGIN - right

    box(slide, right, top, right_w, Inches(1.26))
    bar = box(slide, right, top, Inches(0.05), Inches(1.26), fill=GREEN)
    bar.line.fill.background()
    text(slide, right + Inches(0.28), top + Inches(0.18), right_w - Inches(0.56),
         Inches(0.4), [("Ship epsilon 3", 19, INK, True)])
    text(slide, right + Inches(0.28), top + Inches(0.6), right_w - Inches(0.56),
         Inches(0.52),
         [(f"epsilon {actual:.2f}, delta 0.00001, for each hospital. Accuracy "
           f"{dp3['target_test_auc_mean']:.3f} against {tuned['target_test_auc_mean']:.3f} "
           f"with no privacy at all.", 12.5, MUTED, False)], spacing=1.08)

    rows = [
        ("Careless is not cheap", ORANGE,
         f"The leaking model was also the worst, at "
         f"{overfit['pct_of_bayes_ceiling']:.1%} of the best score."),
        ("Care is not a guarantee", BLUE,
         f"LiRA still scores {tuned_lira:.3f} on the careful model. Only DP reaches "
         "chance."),
        ("Scale makes privacy cheap", VIOLET,
         "Three times the patients cut the cost from 2.5% to 0.6%."),
    ]
    row_top = top + Inches(1.46)
    row_h = Inches(0.86)
    for index, (title, colour, body) in enumerate(rows):
        y = row_top + index * (row_h + Inches(0.14))
        box(slide, right, y, right_w, row_h)
        bar = box(slide, right, y, Inches(0.05), row_h, fill=colour)
        bar.line.fill.background()
        text(slide, right + Inches(0.28), y + Inches(0.12), right_w - Inches(0.56),
             Inches(0.3), [(title, 14.5, INK, True)])
        text(slide, right + Inches(0.28), y + Inches(0.42), right_w - Inches(0.56),
             Inches(0.4), [(body, 12, MUTED, False)], spacing=1.05)

    footer(slide, "The jury asked for scale and repeat runs. We delivered "
                  f"{cohort:,} patients and {n_seeds} independent runs. "
                  "We also report where we could not prove our own case.")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load()

    chart_tradeoff(OUT_DIR / "chart_tradeoff.png")
    chart_subgroups(OUT_DIR / "chart_subgroups.png")

    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    for builder in (slide_title, slide_objective, slide_solution,
                    slide_validation, slide_results):
        builder(prs, data)

    out = OUT_DIR / "unlisted_gisec_final.pptx"
    prs.save(out)
    print(f"wrote {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
