# ConocoPhillips ML Research

This repo contains the original Colab notebook plus a rebuilt, reproducible ERCOT load forecasting workflow.

## What is included

- `ConocoPhillips_ML_Research.ipynb`: the original notebook baseline.
- `ConocoPhillips_ML_Research_Improved.ipynb`: the cleaned-up report notebook.
- `scripts/build_improved_notebook.py`: end-to-end pipeline that rebuilds the local ERCOT cache, trains the models, writes plots and metrics, and regenerates the improved notebook.
- `data/ercot_eia_hourly.csv`: aligned ERCOT actual load and published day-ahead forecast data from the EIA bulk feed.
- `artifacts/`: metrics, plots, rolling-backtest outputs, and test-set predictions from the latest run.

## Modeling direction

The rebuilt workflow reframes the project as a 24-hour-ahead ERCOT load forecasting problem. Instead of only doing blind time-series extrapolation, it also treats the published day-ahead forecast as an operational prior and learns how to improve it.

Best holdout result from the current run:

- Model: `blend_ridge_direct_hgb_direct`
- RMSE: `1362.08`
- Improvement vs published day-ahead forecast: `16.67%`
- Improvement vs naive 24-hour lag: `61.75%`

## Rebuild

Create a Python environment and install the requirements, then run:

```bash
python scripts/build_improved_notebook.py
```

The script will regenerate the cache, metrics, plots, and improved notebook from the local project files.
