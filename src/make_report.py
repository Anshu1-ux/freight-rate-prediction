"""Builds report.pdf (validation approach + December chart). Run after cv.py and train.py."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

TEAL = "#064A56"
df = pd.read_csv("data/train_test.csv", parse_dates=["date"])
df["rpm"] = df.posted_rate / df.distance
ok = df[(df.rpm >= 1) & (df.rpm <= 5)]

# ---- figure 1: split design + drift ------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(8.2, 4.6), dpi=170, gridspec_kw=dict(height_ratios=[1.0, 1.25]))
a = ax[0]; d = lambda s: pd.Timestamp(s)
rows = [("Fold 1", "2025-01-01", "2025-05-01", "2025-07-01"), ("Fold 2", "2025-01-01", "2025-07-01", "2025-09-01"),
        ("Fold 3", "2025-01-01", "2025-09-01", "2025-11-01")]
for i, (n, s, m, e) in enumerate(rows):
    y = 3 - i
    a.barh(y, (d(m) - d(s)).days, left=mdates.date2num(d(s)), color=TEAL, height=.55)
    a.barh(y, (d(e) - d(m)).days, left=mdates.date2num(d(m)), color="#E0A030", height=.55)
a.barh(0, 61, left=mdates.date2num(d("2025-11-01")), color="#C0392B", height=.55)
a.set_yticks([3, 2, 1, 0]); a.set_yticklabels(["Fold 1", "Fold 2", "Fold 3", "Final"]); a.xaxis_date()
a.xaxis.set_major_formatter(mdates.DateFormatter("%b")); a.set_xlim(mdates.date2num(d("2025-01-01")), mdates.date2num(d("2026-01-01")))
a.barh(1e3, 0)  # keep limits sane
a.set_ylim(-.6, 3.6); a.spines[["top", "right"]].set_visible(False)
a.set_title("Expanding-window, time-forward validation (teal = train, amber = test, red = Nov-Dec hold-out to predict)", fontsize=8.2, loc="left")
g = ok.groupby("date").agg(lr=("rpm", lambda s: np.log(s).mean()), mi=("market_index", "mean"))
b = ax[1]; b.plot(g.index, g.lr.rolling(14, center=True).mean(), color=TEAL, lw=2, label="log $/mile level (14-day mean)")
b.set_ylabel("log $/mile", color=TEAL, fontsize=8); b.tick_params(labelsize=7)
b2 = b.twinx(); b2.plot(g.index, g.mi.rolling(14, center=True).mean(), color="#E0A030", lw=2, label="market_index (14-day mean)")
b2.set_ylabel("market_index", color="#B07810", fontsize=8); b2.tick_params(labelsize=7)
b.set_title("Rate level drifts up ~0.7%/month while market_index rises then falls -> needs an explicit trend term", fontsize=8.2, loc="left")
b.spines[["top"]].set_visible(False); b2.spines[["top"]].set_visible(False)
fig.tight_layout(); fig.savefig("scorer_results/report_fig_split_drift.png"); plt.close(fig)

# ---- document ----------------------------------------------------------------------------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor(TEAL), fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor(TEAL), fontSize=11.5, spaceBefore=9, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.4, leading=12.6, spaceAfter=4)
BL = ParagraphStyle("BL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#455A60"))
def bl(t): return Paragraph(t, BL, bulletText="-")

cvs = pd.read_csv("cv_summary.csv", index_col=0); folds = pd.read_csv("cv_hybrid_folds.csv"); un = pd.read_csv("cv_unseen_city.csv", index_col=0)
def tbl(data, widths, hdr=True):
    cell = ParagraphStyle("cell", parent=B, fontSize=8.2, leading=10, spaceAfter=0)
    data = [[Paragraph(str(c), cell) if (isinstance(c, str) and len(c) > 28) else c for c in row] for row in data]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.2), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#9DAFB3")),
          ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F7")])]
    if hdr: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t

S = []
S += [Paragraph("Freight Rate Prediction - Validation Approach &amp; December Forecast", H1),
      Paragraph("Machine Learning Engineer assessment. Model: hybrid robust trend + LightGBM residual, fitted on train_test.csv (48,000 loads, Jan-Oct 2025), applied to validation.csv (12,000 loads, Nov-Dec 2025).", B)]

S += [Paragraph("1. How the data was split and why", H2),
      Paragraph("The deliverable is a forecast of the <b>future</b>: train_test.csv covers 2025-01-01 to 2025-10-31 and validation.csv covers 2025-11-01 to 2025-12-31, so nothing in the final test overlaps in time with training. "
                "A random K-fold split would leak same-day and neighbouring-day market conditions into validation and overstate performance, so I did not use one. Instead:", B),
      bl("<b>Expanding-window, time-forward folds</b> with 2-month test blocks (the same length as the real hold-out): train Jan-Apr / test May-Jun; train Jan-Jun / test Jul-Aug; train Jan-Aug / test Sep-Oct. All model choices (objective, trend term, features) were made on these folds; the final model is refit on all ten months."),
      bl("<b>Unseen-city stress test.</b> 8 of the 72 cities in validation never appear in train_test.csv (about 12% of validation loads touch one). I repeatedly held out 8 random cities from training, then scored them on Sep-Oct. Error on unseen-city loads was 1.94% vs 1.86% on seen-city loads, so the coordinate-based features generalise to new cities."),
      bl("<b>Noisy labels handled in evaluation.</b> About 1.3% of labels look corrupt (0.7% with rate per mile above $5, typically 3-4x the norm, and 0.7% below $1) and they dominate RMSE/R<super>2</super>. I report metrics on all rows <i>and</i> on credible rows, so model quality is not hidden by label noise nobody can predict."),
      Spacer(1, 4), Image("scorer_results/report_fig_split_drift.png", width=6.3 * inch, height=3.53 * inch)]

S += [Paragraph("2. Data-quality issues found and how they were handled", H2),
      tbl([["Issue (train / validation)", "Handling"],
           ["Negative weights (292 / 145 rows); magnitude matches valid loads", "Treated as sign errors: abs(weight), plus a flag feature"],
           ["Weight hard-capped at 47,500 lb (1,191 / 297 rows)", "Censored values kept, flagged with weight_capped"],
           ["Missing weight (300 / 165)", "Left as NaN (LightGBM native) + flag; stage-1 fill with median"],
           ["Missing market_index (374 / 249)", "Filled with that day's median across loads (daily signal, in-day sd ~0.025); flag kept"],
           ["Label outliers: $/mile &gt; 5 or &lt; 1 (646 rows, 1.3%)", "Excluded from fitting only; L1 loss in the boosted stage for robustness"],
           ["Distance vs. coordinates", "Verified consistent (distance / haversine ~1.19, per-route jitter about 2%); no fix needed"],
           ["Unseen cities in validation; train/val date shift", "No city one-hot / target encoding; use coordinates, distance, equipment; explicit time trend"]],
          [3.3 * inch, 3.6 * inch])]

S += [Paragraph("3. Model and key results", H2),
      Paragraph("Key finding: log(rate per mile) depends on distance, equipment, market_index and quote_signal, but also on a <b>steady upward drift (~0.7% per month)</b> that market_index does not explain. "
                "Tree models cannot extrapolate it: a plain LightGBM holds rates at the last level it has seen, which matters because Nov-Dec is entirely outside the training dates. The chosen model is therefore two-stage: "
                "(1) a Huber-robust linear model on log $/mile with a linear time term, log market_index, quote_signal, distance terms and equipment; "
                "(2) LightGBM with an L1 objective on stage-1 residuals using route geometry, equipment, weight, weekday and data-quality flags, with no calendar-level features. Prediction = exp(stage 1 + stage 2) x distance.", B)]
lab = {"Naive $/mile by equipment": "Naive $/mile by equipment", "Plain LightGBM (L1, log target)": "Plain LightGBM (L1)", "Hybrid trend + LightGBM residual": "Hybrid (chosen)"}
data = [["Model (mean of 3 folds)", "MAE $", "MAPE %", "WAPE %", "R2", "MAE $ (clean)", "MAPE % (clean)"]]
for k, r in cvs.iterrows(): data.append([lab[k], f"{r.MAE:.0f}", f"{r.MAPE:.2f}", f"{r.WAPE:.2f}", f"{r.R2:.3f}", f"{r.MAE_clean:.0f}", f"{r.MAPE_clean:.2f}"])
S += [tbl(data, [2.0 * inch, .65 * inch, .7 * inch, .7 * inch, .55 * inch, .95 * inch, 1.0 * inch]), Spacer(1, 8)]
data = [["Hybrid per fold", "MAE $", "MAPE %", "R2", "MAPE % (clean)", "Bias (log, clean)"]]
for _, r in folds.iterrows(): data.append([r.fold, f"{r.MAE:.0f}", f"{r.MAPE:.2f}", f"{r.R2:.3f}", f"{r.MAPE_clean:.2f}", f"{r.bias_clean:+.3f}"])
S += [tbl(data, [1.9 * inch, .8 * inch, .8 * inch, .7 * inch, 1.3 * inch, 1.3 * inch]),
      Paragraph("R<super>2</super> is capped near 0.83 on all rows by the corrupt labels; on credible rows the hybrid's typical error is about 2.3%. The weakest fold (Jul-Aug, -3% bias) coincides with the market_index peak; a possible cause (not tested) is a nonlinear pass-through of the index into rates, which would be the first thing to investigate next.", SM)]

S += [Paragraph("4. Fixed December 2025 prediction chart (produced by score.py)", H2),
      Paragraph("Chart inputs contain only the date, but the model also needs the two daily market features. I used the December <b>daily median market_index taken from validation.csv input features</b> (no labels) and the December median quote_signal (load-level noise with no daily structure). "
                "The result is the expected weekly cycle (midweek peak, weekend dip) on the upward trend, about $2.33/mile for a 360-mile Dry Van load ($828-$851).", B),
      Image("scorer_results/candidate_december.png", width=5.6 * inch, height=5.6 * inch * 0.3257)]

SimpleDocTemplate("report.pdf", pagesize=letter, leftMargin=.75 * inch, rightMargin=.75 * inch, topMargin=.65 * inch, bottomMargin=.65 * inch,
                  title="Freight Rate Prediction - Validation Report").build(S)
print("wrote report.pdf")
