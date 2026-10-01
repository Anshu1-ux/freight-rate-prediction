#!/usr/bin/env bash
set -euo pipefail
python src/cv.py
python src/train.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
python src/make_report.py
