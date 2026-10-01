# Freight Rate Prediction

Hybrid model (robust trend model + LightGBM residual) predicting `posted_rate` for truckload freight.
Developed on `data/train_test.csv` (Jan-Oct 2025), applied to `data/validation.csv` (Nov-Dec 2025).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` runs, in order:

| Step | Command | Output |
|---|---|---|
| Validation study | `python src/cv.py` | `cv_summary.csv`, `cv_hybrid_folds.csv`, `cv_unseen_city.csv` |
| Final fit + predictions | `python src/train.py` | `validation_predictions.csv`, `december_predictions.csv` |
| Official scorer | `python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv` | `scorer_results/candidate_december.png` |
| Report | `python src/make_report.py` | `report.pdf` |

Python 3.10+ (developed on 3.12). Runtime is about 1-2 minutes on a laptop CPU.

## Layout

- `src/features.py`: cleaning (negative/capped/missing weight, missing `market_index`) and features
- `src/model.py`: `HybridRateModel` (stage 1 Huber linear trend model, stage 2 LightGBM L1 on residuals)
- `src/cv.py`: time-forward CV, baselines, unseen-city stress test
- `src/train.py`: final fit, validation + December predictions
- `src/make_report.py`: builds the PDF report
- `score.py`: provided scorer (unchanged)

## Approach in brief

- **Split**: expanding-window, time-forward folds with 2-month test blocks, mirroring the Nov-Dec hold-out. No random K-fold (it leaks same-day market state). Extra stress test holding out 8 cities, since 8 validation cities never appear in training.
- **Data quality**: negative weights sign-flipped, 47,500 lb cap flagged, missing weight flagged, missing `market_index` filled from that day's median, ~1.3% corrupt labels ($/mile outside [1, 5]) excluded from fitting only.
- **Key finding**: log $/mile drifts up ~0.7%/month beyond what `market_index` explains. Trees can't extrapolate this, so a linear time term lives in stage 1 and the trees use no calendar-level features.
- **Result** (mean of 3 folds, credible rows): MAPE 2.34% vs 2.86% for a plain LightGBM and 8.37% for a naive $/mile baseline.

## December chart assumption

`december_chart_inputs.csv` only has a date, but the model also uses `market_index` and `quote_signal`.
The chart uses the December daily median `market_index` from the input features in `validation.csv`
(no labels) and the December median `quote_signal`. This is why the chart has a weekly cycle.

## Data

The `data/` folder is not committed. Place the provided files there before running:
`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`.
