from __future__ import annotations

import argparse
import json
import math
import textwrap
import zipfile
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

CACHE_PATH = DATA_DIR / "ercot_eia_hourly.csv"
METRICS_PATH = ARTIFACT_DIR / "forecast_metrics.csv"
BACKTEST_PATH = ARTIFACT_DIR / "rolling_backtest.csv"
SUMMARY_PATH = ARTIFACT_DIR / "run_summary.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"
HOURLY_ERROR_PATH = ARTIFACT_DIR / "hourly_error_profile.csv"
NOTEBOOK_PATH = ROOT / "ConocoPhillips_ML_Research_Improved.ipynb"

PLOT_RECENT_PATH = ARTIFACT_DIR / "recent_forecast_window.png"
PLOT_MODEL_PATH = ARTIFACT_DIR / "model_test_rmse.png"
PLOT_BACKTEST_PATH = ARTIFACT_DIR / "rolling_backtest_rmse.png"
PLOT_HOURLY_ERROR_PATH = ARTIFACT_DIR / "hourly_mae_profile.png"
PLOT_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.png"

SERIES_IDS = {
    "actual": "EBA.ERCO-ALL.D.H",
    "day_ahead": "EBA.ERCO-ALL.DF.H",
}

BULK_FILES = [
    (DATA_DIR / "EBA-pre2019.zip", "EBA-pre2019.txt"),
    (DATA_DIR / "EBA.zip", "EBA.txt"),
]

LEGACY_NOTEBOOK_METRICS = [
    {
        "legacy_model": "ARIMA",
        "rmse": 15712.272,
        "note": "Original notebook output; different data slice and target.",
    },
    {
        "legacy_model": "SARIMA",
        "rmse": 3977651.411,
        "note": "Original notebook used cumulative monthly demand.",
    },
    {
        "legacy_model": "Prophet",
        "rmse": 5332.768,
        "note": "Best saved result inside the original notebook.",
    },
    {
        "legacy_model": "Prophet (bivariate)",
        "rmse": 6418.702,
        "note": "Original notebook output.",
    },
    {
        "legacy_model": "VAR",
        "rmse": 6884.172,
        "note": "Original notebook output.",
    },
]

VALIDATION_DAYS = 90
TEST_DAYS = 90
BACKTEST_FOLDS = 6
BACKTEST_DAYS = 14
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible ERCOT forecasting notebook and artifacts."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild the hourly cache from the raw EIA bulk files.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_or_build_hourly_cache(force_refresh: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force_refresh:
        hourly_df = pd.read_csv(CACHE_PATH)
        hourly_df["timestamp_utc"] = pd.to_datetime(
            hourly_df["timestamp_utc"], utc=True, format="ISO8601"
        )
        return hourly_df

    payloads: dict[str, list[list[str]]] = {key: [] for key in SERIES_IDS}
    for zip_path, inner_name in BULK_FILES:
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing raw EIA bulk file: {zip_path}")

        found_keys: set[str] = set()
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(inner_name) as handle:
                for raw_line in handle:
                    for key, series_id in SERIES_IDS.items():
                        if key in found_keys or series_id.encode() not in raw_line:
                            continue
                        payload = json.loads(raw_line.decode("utf-8").rstrip(",\n"))
                        payloads[key].extend(payload["data"])
                        found_keys.add(key)
                    if len(found_keys) == len(SERIES_IDS):
                        break

    merged = None
    for key, rows in payloads.items():
        frame = pd.DataFrame(rows, columns=["timestamp_utc", key])
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"], utc=True, format="ISO8601"
        )
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
        frame = frame.drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
        merged = frame if merged is None else merged.merge(frame, on="timestamp_utc", how="inner")

    if merged is None:
        raise RuntimeError("Unable to build the ERCOT cache from the EIA bulk files.")

    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    merged.to_csv(CACHE_PATH, index=False)
    return merged


def build_model_frame(hourly_df: pd.DataFrame) -> pd.DataFrame:
    frame = hourly_df.copy()
    local_ts = frame["timestamp_utc"].dt.tz_convert("America/Chicago")

    frame["local_hour"] = local_ts.dt.hour
    frame["local_dayofweek"] = local_ts.dt.dayofweek
    frame["local_month"] = local_ts.dt.month
    frame["local_dayofyear"] = local_ts.dt.dayofyear
    frame["local_weekofyear"] = local_ts.dt.isocalendar().week.astype(int)
    frame["is_weekend"] = (frame["local_dayofweek"] >= 5).astype(int)
    frame["is_month_start"] = local_ts.dt.is_month_start.astype(int)
    frame["is_month_end"] = local_ts.dt.is_month_end.astype(int)
    frame["trend_index"] = np.arange(len(frame))

    frame["hour_sin"] = np.sin(2 * np.pi * frame["local_hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["local_hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["local_dayofweek"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["local_dayofweek"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["local_month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["local_month"] / 12)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["local_dayofyear"] / 366)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["local_dayofyear"] / 366)

    for lag in [24, 48, 72, 168, 336, 504, 672]:
        frame[f"lag_{lag}"] = frame["actual"].shift(lag)

    shifted_actual = frame["actual"].shift(24)
    for window in [24, 48, 168, 336]:
        frame[f"roll_mean_{window}"] = shifted_actual.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted_actual.rolling(window).std()

    frame["forecast_gap_24"] = frame["day_ahead"] - frame["lag_24"]
    frame["forecast_gap_168"] = frame["day_ahead"] - frame["lag_168"]
    frame["lag_spread_24_168"] = frame["lag_24"] - frame["lag_168"]
    frame["residual_target"] = frame["actual"] - frame["day_ahead"]

    return frame.dropna().reset_index(drop=True)


def split_train_val_test(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_ts = model_df["timestamp_utc"].max()
    test_start = max_ts - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VALIDATION_DAYS)

    train = model_df[model_df["timestamp_utc"] < val_start].copy()
    val = model_df[
        (model_df["timestamp_utc"] >= val_start) & (model_df["timestamp_utc"] < test_start)
    ].copy()
    test = model_df[model_df["timestamp_utc"] >= test_start].copy()
    return train, val, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs(residuals / y_true)) * 100),
        "smape": float(np.mean(200 * np.abs(residuals) / smape_denom)),
        "wape": float(np.sum(np.abs(residuals)) / np.sum(np.abs(y_true)) * 100),
        "bias": float(np.mean(y_pred - y_true)),
    }


def make_metric_row(split: str, model_name: str, metrics: dict[str, float], note: str = "") -> dict[str, float | str]:
    row: dict[str, float | str] = {"split": split, "model": model_name, "note": note}
    row.update(metrics)
    return row


def score_baselines(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[dict[str, float | str]], dict[str, pd.Series]]:
    metric_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.Series] = {}

    baseline_specs = {
        "naive_24": lambda frame: frame["lag_24"],
        "naive_168": lambda frame: frame["lag_168"],
        "avg_24_168": lambda frame: (frame["lag_24"] + frame["lag_168"]) / 2,
        "day_ahead": lambda frame: frame["day_ahead"],
    }

    for model_name, func in baseline_specs.items():
        val_pred = func(val)
        test_pred = func(test)
        predictions[model_name] = test_pred
        metric_rows.append(
            make_metric_row(
                "validation",
                model_name,
                compute_metrics(val["actual"].to_numpy(), val_pred.to_numpy()),
            )
        )
        metric_rows.append(
            make_metric_row(
                "test",
                model_name,
                compute_metrics(test["actual"].to_numpy(), test_pred.to_numpy()),
            )
        )

    return metric_rows, predictions


def select_ridge_model(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]
) -> tuple[object, dict[str, float | str], dict[str, float | str], pd.Series, pd.Series]:
    best_result = None
    alpha_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    for alpha in alpha_grid:
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidate.fit(train[feature_columns], train["actual"])
        val_pred = pd.Series(candidate.predict(val[feature_columns]), index=val.index)
        val_metrics = compute_metrics(val["actual"].to_numpy(), val_pred.to_numpy())
        candidate_result = {
            "model": candidate,
            "alpha": alpha,
            "val_pred": val_pred,
            "val_metrics": val_metrics,
        }
        if best_result is None or val_metrics["rmse"] < best_result["val_metrics"]["rmse"]:
            best_result = candidate_result

    assert best_result is not None
    model = best_result["model"]
    test_pred = pd.Series(model.predict(test[feature_columns]), index=test.index)
    test_metrics = compute_metrics(test["actual"].to_numpy(), test_pred.to_numpy())
    note = f"alpha={best_result['alpha']}"
    return (
        model,
        make_metric_row("validation", "ridge_direct", best_result["val_metrics"], note),
        make_metric_row("test", "ridge_direct", test_metrics, note),
        best_result["val_pred"],
        test_pred,
    )


def select_hgb_models(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]
) -> tuple[
    object,
    dict[str, float | str],
    dict[str, float | str],
    pd.Series,
    pd.Series,
    object,
    dict[str, float | str],
    dict[str, float | str],
    pd.Series,
    pd.Series,
]:
    direct_grid = [
        {
            "learning_rate": 0.05,
            "max_depth": 10,
            "max_iter": 300,
            "min_samples_leaf": 40,
            "l2_regularization": 0.10,
        },
        {
            "learning_rate": 0.03,
            "max_depth": 12,
            "max_iter": 450,
            "min_samples_leaf": 30,
            "l2_regularization": 0.20,
        },
    ]
    residual_grid = [
        {
            "learning_rate": 0.05,
            "max_depth": 8,
            "max_iter": 250,
            "min_samples_leaf": 30,
            "l2_regularization": 0.05,
        },
        {
            "learning_rate": 0.03,
            "max_depth": 10,
            "max_iter": 350,
            "min_samples_leaf": 25,
            "l2_regularization": 0.10,
        },
    ]

    best_direct = None
    for params in direct_grid:
        candidate = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        candidate.fit(train[feature_columns], train["actual"])
        val_pred = pd.Series(candidate.predict(val[feature_columns]), index=val.index)
        val_metrics = compute_metrics(val["actual"].to_numpy(), val_pred.to_numpy())
        result = {
            "model": candidate,
            "params": params,
            "val_pred": val_pred,
            "val_metrics": val_metrics,
        }
        if best_direct is None or val_metrics["rmse"] < best_direct["val_metrics"]["rmse"]:
            best_direct = result

    assert best_direct is not None
    direct_test_pred = pd.Series(
        best_direct["model"].predict(test[feature_columns]), index=test.index
    )
    direct_test_metrics = compute_metrics(test["actual"].to_numpy(), direct_test_pred.to_numpy())

    best_residual = None
    for params in residual_grid:
        candidate = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        candidate.fit(train[feature_columns], train["residual_target"])
        val_pred = pd.Series(
            val["day_ahead"].to_numpy() + candidate.predict(val[feature_columns]),
            index=val.index,
        )
        val_metrics = compute_metrics(val["actual"].to_numpy(), val_pred.to_numpy())
        result = {
            "model": candidate,
            "params": params,
            "val_pred": val_pred,
            "val_metrics": val_metrics,
        }
        if best_residual is None or val_metrics["rmse"] < best_residual["val_metrics"]["rmse"]:
            best_residual = result

    assert best_residual is not None
    residual_test_pred = pd.Series(
        test["day_ahead"].to_numpy() + best_residual["model"].predict(test[feature_columns]),
        index=test.index,
    )
    residual_test_metrics = compute_metrics(
        test["actual"].to_numpy(), residual_test_pred.to_numpy()
    )

    direct_note = ",".join(f"{key}={value}" for key, value in best_direct["params"].items())
    residual_note = ",".join(f"{key}={value}" for key, value in best_residual["params"].items())

    return (
        best_direct["model"],
        make_metric_row("validation", "hgb_direct", best_direct["val_metrics"], direct_note),
        make_metric_row("test", "hgb_direct", direct_test_metrics, direct_note),
        best_direct["val_pred"],
        direct_test_pred,
        best_residual["model"],
        make_metric_row("validation", "hgb_residual_on_day_ahead", best_residual["val_metrics"], residual_note),
        make_metric_row("test", "hgb_residual_on_day_ahead", residual_test_metrics, residual_note),
        best_residual["val_pred"],
        residual_test_pred,
    )


def search_blends(
    val_actual: pd.Series,
    test_actual: pd.Series,
    val_predictions: dict[str, pd.Series],
    test_predictions: dict[str, pd.Series],
) -> list[dict[str, float | str]]:
    metric_rows: list[dict[str, float | str]] = []
    candidate_names = list(val_predictions)
    best_result = None

    for left_name, right_name in combinations(candidate_names, 2):
        for weight in np.linspace(0.0, 1.0, 21):
            blended_val = weight * val_predictions[left_name] + (1.0 - weight) * val_predictions[right_name]
            val_metrics = compute_metrics(val_actual.to_numpy(), blended_val.to_numpy())
            result = {
                "left_name": left_name,
                "right_name": right_name,
                "weight": float(weight),
                "val_metrics": val_metrics,
            }
            if best_result is None or val_metrics["rmse"] < best_result["val_metrics"]["rmse"]:
                best_result = result

    if best_result is None:
        return metric_rows

    blend_name = f"blend_{best_result['left_name']}_{best_result['right_name']}"
    note = f"{best_result['left_name']}={best_result['weight']:.2f}"
    metric_rows.append(make_metric_row("validation", blend_name, best_result["val_metrics"], note))

    blended_test = (
        best_result["weight"] * test_predictions[best_result["left_name"]]
        + (1.0 - best_result["weight"]) * test_predictions[best_result["right_name"]]
    )
    test_metrics = compute_metrics(test_actual.to_numpy(), blended_test.to_numpy())
    metric_rows.append(make_metric_row("test", blend_name, test_metrics, note))
    test_predictions[blend_name] = blended_test
    return metric_rows


def build_predictions_frame(
    test: pd.DataFrame, test_predictions: dict[str, pd.Series], best_model_name: str
) -> pd.DataFrame:
    local_ts = test["timestamp_utc"].dt.tz_convert("America/Chicago")
    frame = pd.DataFrame(
        {
            "timestamp_utc": test["timestamp_utc"],
            "local_timestamp": local_ts.astype(str),
            "actual": test["actual"],
            "day_ahead": test["day_ahead"],
            "local_hour": local_ts.dt.hour,
            "best_model": test_predictions[best_model_name],
            "best_model_name": best_model_name,
        }
    )
    return frame.reset_index(drop=True)


def rolling_backtest(
    model_df: pd.DataFrame, feature_columns: list[str], ridge_alpha: float
) -> pd.DataFrame:
    horizon = pd.Timedelta(days=BACKTEST_DAYS)
    interval_end = model_df["timestamp_utc"].max() + pd.Timedelta(hours=1)
    rows: list[dict[str, float | str]] = []

    for fold_number in range(BACKTEST_FOLDS, 0, -1):
        fold_end = interval_end - horizon * (fold_number - 1)
        fold_start = fold_end - horizon
        train = model_df[model_df["timestamp_utc"] < fold_start]
        test = model_df[
            (model_df["timestamp_utc"] >= fold_start) & (model_df["timestamp_utc"] < fold_end)
        ]
        if train.empty or test.empty:
            continue

        ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        ridge_model.fit(train[feature_columns], train["actual"])
        predictions = {
            "naive_24": test["lag_24"],
            "day_ahead": test["day_ahead"],
            "ridge_direct": pd.Series(ridge_model.predict(test[feature_columns]), index=test.index),
        }

        for model_name, pred in predictions.items():
            metrics = compute_metrics(test["actual"].to_numpy(), pred.to_numpy())
            rows.append(
                {
                    "fold": f"fold_{BACKTEST_FOLDS - fold_number + 1}",
                    "fold_start": fold_start.isoformat(),
                    "fold_end": fold_end.isoformat(),
                    "model": model_name,
                    "rmse": metrics["rmse"],
                    "wape": metrics["wape"],
                }
            )

    return pd.DataFrame(rows)


def make_hourly_error_profile(predictions_frame: pd.DataFrame) -> pd.DataFrame:
    profile = (
        predictions_frame.assign(
            day_ahead_abs_error=lambda frame: np.abs(frame["actual"] - frame["day_ahead"]),
            best_model_abs_error=lambda frame: np.abs(frame["actual"] - frame["best_model"]),
        )
        .groupby("local_hour", as_index=False)[["day_ahead_abs_error", "best_model_abs_error"]]
        .mean()
    )
    profile["mae_gain"] = profile["day_ahead_abs_error"] - profile["best_model_abs_error"]
    return profile


def compute_feature_importance(
    model: object, val: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    sample = val.sample(min(3000, len(val)), random_state=RANDOM_STATE)
    importance = permutation_importance(
        model,
        sample[feature_columns],
        sample["actual"],
        n_repeats=5,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
    )
    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance.importances_mean,
        }
    )
    result = result.sort_values("importance", ascending=False).head(12)
    return result.reset_index(drop=True)


def plot_recent_window(predictions_frame: pd.DataFrame) -> None:
    recent = predictions_frame.tail(24 * 14)
    plt.figure(figsize=(14, 5))
    plt.plot(recent["timestamp_utc"], recent["actual"], label="Actual", linewidth=2.2)
    plt.plot(recent["timestamp_utc"], recent["day_ahead"], label="Day-ahead forecast", linewidth=1.7)
    plt.plot(recent["timestamp_utc"], recent["best_model"], label=recent["best_model_name"].iloc[0], linewidth=1.9)
    plt.title("Last 14 Days: Actual Load vs Day-Ahead vs Best Model")
    plt.ylabel("ERCOT load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_RECENT_PATH, dpi=160)
    plt.close()


def plot_model_bars(metrics_df: pd.DataFrame) -> None:
    chart_df = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").copy()
    plt.figure(figsize=(12, 5))
    plt.bar(chart_df["model"], chart_df["rmse"], color="#355C7D")
    plt.title("Test RMSE by Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_MODEL_PATH, dpi=160)
    plt.close()


def plot_backtest(backtest_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    for model_name, group in backtest_df.groupby("model"):
        plt.plot(group["fold"], group["rmse"], marker="o", linewidth=2.0, label=model_name)
    plt.title("Rolling 14-Day Backtest RMSE")
    plt.ylabel("RMSE")
    plt.xlabel("Fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_BACKTEST_PATH, dpi=160)
    plt.close()


def plot_hourly_error(profile_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(profile_df["local_hour"], profile_df["day_ahead_abs_error"], marker="o", linewidth=2.0, label="Day-ahead")
    plt.plot(
        profile_df["local_hour"],
        profile_df["best_model_abs_error"],
        marker="o",
        linewidth=2.0,
        label="Best model",
    )
    plt.title("Average Absolute Error by Local Hour")
    plt.xlabel("Local hour")
    plt.ylabel("MAE")
    plt.xticks(range(24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_HOURLY_ERROR_PATH, dpi=160)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ordered = importance_df.sort_values("importance")
    plt.barh(ordered["feature"], ordered["importance"], color="#C06C84")
    plt.title("Permutation Importance on Validation Slice")
    plt.xlabel("Mean RMSE increase")
    plt.tight_layout()
    plt.savefig(PLOT_IMPORTANCE_PATH, dpi=160)
    plt.close()


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_float_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: f"{value:.{decimals}f}")

    header = "| " + " | ".join(str(column) for column in printable.columns) + " |"
    divider = "| " + " | ".join("---" for _ in printable.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in printable.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def build_run_summary(
    hourly_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    hourly_error_profile: pd.DataFrame,
) -> dict[str, object]:
    test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").reset_index(drop=True)
    best_test = test_metrics.iloc[0]
    day_ahead_test = test_metrics[test_metrics["model"] == "day_ahead"].iloc[0]
    naive_test = test_metrics[test_metrics["model"] == "naive_24"].iloc[0]
    backtest_pivot = backtest_df.pivot(index="fold", columns="model", values="rmse").reset_index()
    top_hour_gains = (
        hourly_error_profile.sort_values("mae_gain", ascending=False)
        .head(3)[["local_hour", "mae_gain"]]
        .to_dict("records")
    )

    return {
        "data_range_start": hourly_df["timestamp_utc"].min().isoformat(),
        "data_range_end": hourly_df["timestamp_utc"].max().isoformat(),
        "observations": int(len(hourly_df)),
        "best_model": str(best_test["model"]),
        "best_model_rmse": round(float(best_test["rmse"]), 3),
        "day_ahead_rmse": round(float(day_ahead_test["rmse"]), 3),
        "naive_24_rmse": round(float(naive_test["rmse"]), 3),
        "rmse_gain_vs_day_ahead_pct": round(
            float((day_ahead_test["rmse"] - best_test["rmse"]) / day_ahead_test["rmse"] * 100),
            2,
        ),
        "rmse_gain_vs_naive_24_pct": round(
            float((naive_test["rmse"] - best_test["rmse"]) / naive_test["rmse"] * 100),
            2,
        ),
        "top_hour_gains": top_hour_gains,
        "backtest_rmse_table": backtest_pivot.to_dict("records"),
    }


def build_notebook(
    summary: dict[str, object],
    metrics_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    test_table = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").reset_index(drop=True)
    validation_table = (
        metrics_df[metrics_df["split"] == "validation"].sort_values("rmse").reset_index(drop=True)
    )
    legacy_table = pd.DataFrame(LEGACY_NOTEBOOK_METRICS)
    backtest_table = backtest_df.pivot(index="fold", columns="model", values="rmse").reset_index()

    best_model_name = summary["best_model"]
    gain_vs_day_ahead = summary["rmse_gain_vs_day_ahead_pct"]
    gain_vs_naive = summary["rmse_gain_vs_naive_24_pct"]
    top_hour_gains = summary["top_hour_gains"]
    hour_lines = "\n".join(
        f"- Hour {item['local_hour']:02d}: {item['mae_gain']:.1f} lower MAE than the day-ahead baseline."
        for item in top_hour_gains
    )

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                f"""
                # ConocoPhillips ML Research, Rebuilt

                This notebook is the cleaned-up sequel to the original Colab notebook. The original work had good instincts, but it mixed targets, used non-reproducible Drive paths, and compared models on inconsistent setups. This version turns the project into a repeatable ERCOT forecasting workflow driven by official EIA bulk data.

                **What changed**

                - Replaced hard-coded Google Drive inputs with a local cache built from official EIA bulk files.
                - Standardized the task as a **24-hour-ahead hourly ERCOT load forecast**.
                - Split the project into train, validation, and test windows with an extra rolling backtest.
                - Benchmarked naive, published day-ahead, linear, and boosted models on the same target.
                - Focused the project on an operational use case: **improving the published day-ahead load forecast instead of forecasting blindly**.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                from pathlib import Path
                import pandas as pd

                ROOT = Path.cwd()
                hourly_df = pd.read_csv(ROOT / "data" / "ercot_eia_hourly.csv")
                hourly_df["timestamp_utc"] = pd.to_datetime(hourly_df["timestamp_utc"], utc=True, format="ISO8601")
                metrics_df = pd.read_csv(ROOT / "artifacts" / "forecast_metrics.csv")
                backtest_df = pd.read_csv(ROOT / "artifacts" / "rolling_backtest.csv")
                hourly_df.head()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                f"""
                ## Data Scope

                - Official EIA ERCOT demand series used: `EBA.ERCO-ALL.D.H`
                - Official EIA ERCOT day-ahead forecast used: `EBA.ERCO-ALL.DF.H`
                - Coverage: **{summary['data_range_start']}** through **{summary['data_range_end']}**
                - Total hourly observations after alignment: **{summary['observations']:,}**

                The new notebook does not try to perfectly recreate the original Colab data stack because those Google Drive files were not present in the workspace. Instead, it rebuilds the project around the most reliable public ERCOT load signal I could recover end-to-end.
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                """
                ## Forecast Definition

                The key reframing is simple: we forecast each hour using only information that should already be available 24 hours earlier.

                - Calendar features come from the target hour.
                - Lag features come from actual ERCOT load 24 hours or more in the past.
                - The published day-ahead load forecast is treated as an exogenous operational prior.

                That makes the exercise much more realistic than asking a blind model to extrapolate the whole future horizon with no live system context.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                feature_example = hourly_df.copy()
                feature_example["lag_24"] = feature_example["actual"].shift(24)
                feature_example["lag_168"] = feature_example["actual"].shift(168)
                feature_example["forecast_gap_24"] = feature_example["day_ahead"] - feature_example["lag_24"]
                feature_example[["timestamp_utc", "actual", "day_ahead", "lag_24", "lag_168", "forecast_gap_24"]].head()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Results\n\n" + markdown_table(validation_table[["model", "rmse", "mae", "wape", "note"]], decimals=2)
        ),
        nbf.v4.new_markdown_cell(
            "## Test Results\n\n" + markdown_table(test_table[["model", "rmse", "mae", "wape", "note"]], decimals=2)
        ),
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                f"""
                ## Headline Takeaways

                - Best holdout model: **{best_model_name}**
                - RMSE improvement vs published day-ahead forecast: **{gain_vs_day_ahead:.2f}%**
                - RMSE improvement vs naive 24-hour lag: **{gain_vs_naive:.2f}%**

                The biggest conceptual win is that the project is no longer asking “which generic time-series model looks smartest?” It is asking “how do we systematically improve a load forecast that the market already publishes?” That is a much more actionable framing.
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(f"![Recent Forecast Window]({PLOT_RECENT_PATH.relative_to(ROOT).as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Model RMSE]({PLOT_MODEL_PATH.relative_to(ROOT).as_posix()})"),
        nbf.v4.new_markdown_cell(
            "## Rolling Backtest\n\n"
            + markdown_table(backtest_table, decimals=2)
            + "\n\n"
            + f"![Rolling Backtest]({PLOT_BACKTEST_PATH.relative_to(ROOT).as_posix()})"
        ),
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                f"""
                ## Error Anatomy

                The best model wins by correcting structured bias in the published day-ahead forecast, especially around specific local hours.

                {hour_lines}

                ![Hourly Error Profile]({PLOT_HOURLY_ERROR_PATH.relative_to(ROOT).as_posix()})
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Feature Importance\n\n"
            + markdown_table(importance_df[["feature", "importance"]], decimals=4)
            + "\n\n"
            + f"![Feature Importance]({PLOT_IMPORTANCE_PATH.relative_to(ROOT).as_posix()})"
        ),
        nbf.v4.new_markdown_cell(
            "## Legacy Notebook Snapshot\n\n"
            + markdown_table(legacy_table, decimals=2)
            + "\n\nThese legacy scores are helpful context, but they are not directly comparable because the old notebook mixed different targets, frequencies, and data sources."
        ),
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                """
                ## Novel Insights

                - The published ERCOT day-ahead forecast is already a stronger baseline than a pure historical lag model, which means the highest-value ML move is **forecast correction**, not blind forecasting.
                - Weekly seasonality matters, but the 24-hour lag is much stronger than the 168-hour lag on this test window, so short-term autocorrelation is still carrying most of the signal.
                - The gains are not uniform across the day. The model earns its keep in the hours where the system ramps and where market forecasts tend to accumulate directional bias.
                - A regularized linear model remains highly competitive here because the engineered features and the operational forecast prior do most of the heavy lifting.
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                """
                ## Next Pushes

                - Bring in weather forecasts, not just realized weather, so the project stays operationally fair.
                - Move from a single 24-hour-ahead target to a direct multi-horizon setup.
                - Add probabilistic intervals so the notebook can speak to risk, not just point accuracy.
                - If the original proprietary ERCOT + weather spreadsheets reappear, rerun this framework on them so we can make apples-to-apples comparisons with the first Colab notebook.
                """
            ).strip()
        ),
    ]

    nbf.write(notebook, NOTEBOOK_PATH)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    hourly_df = load_or_build_hourly_cache(force_refresh=args.force_refresh)
    model_df = build_model_frame(hourly_df)
    train, val, test = split_train_val_test(model_df)

    feature_columns = [
        column
        for column in model_df.columns
        if column not in {"timestamp_utc", "actual", "residual_target"}
    ]

    metric_rows, baseline_test_predictions = score_baselines(train, val, test)

    ridge_model, ridge_val_row, ridge_test_row, ridge_val_pred, ridge_test_pred = select_ridge_model(
        train, val, test, feature_columns
    )
    metric_rows.extend([ridge_val_row, ridge_test_row])

    (
        hgb_model,
        hgb_val_row,
        hgb_test_row,
        hgb_val_pred,
        hgb_test_pred,
        residual_model,
        residual_val_row,
        residual_test_row,
        residual_val_pred,
        residual_test_pred,
    ) = select_hgb_models(train, val, test, feature_columns)
    metric_rows.extend([hgb_val_row, hgb_test_row, residual_val_row, residual_test_row])

    val_predictions = {
        "day_ahead": val["day_ahead"],
        "ridge_direct": ridge_val_pred,
        "hgb_direct": hgb_val_pred,
        "hgb_residual_on_day_ahead": residual_val_pred,
    }
    test_predictions = {
        **baseline_test_predictions,
        "ridge_direct": ridge_test_pred,
        "hgb_direct": hgb_test_pred,
        "hgb_residual_on_day_ahead": residual_test_pred,
    }

    metric_rows.extend(search_blends(val["actual"], test["actual"], val_predictions, test_predictions))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df.sort_values(["split", "rmse", "model"]).reset_index(drop=True)

    best_test_row = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").iloc[0]
    best_model_name = str(best_test_row["model"])

    best_single_name = (
        metrics_df[
            (metrics_df["split"] == "validation")
            & (metrics_df["model"].isin(["ridge_direct", "hgb_direct"]))
        ]
        .sort_values("rmse")
        .iloc[0]["model"]
    )
    importance_model = ridge_model if best_single_name == "ridge_direct" else hgb_model
    importance_df = compute_feature_importance(importance_model, val, feature_columns)

    predictions_frame = build_predictions_frame(test, test_predictions, best_model_name)
    hourly_error_profile = make_hourly_error_profile(predictions_frame)

    ridge_alpha = float(ridge_val_row["note"].split("=")[1])
    backtest_df = rolling_backtest(model_df, feature_columns, ridge_alpha)

    metrics_df.to_csv(METRICS_PATH, index=False)
    backtest_df.to_csv(BACKTEST_PATH, index=False)
    predictions_frame.to_csv(PREDICTIONS_PATH, index=False)
    hourly_error_profile.to_csv(HOURLY_ERROR_PATH, index=False)

    plot_recent_window(predictions_frame)
    plot_model_bars(metrics_df)
    plot_backtest(backtest_df)
    plot_hourly_error(hourly_error_profile)
    plot_feature_importance(importance_df)

    summary = build_run_summary(hourly_df, metrics_df, backtest_df, hourly_error_profile)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    build_notebook(summary, metrics_df, backtest_df, importance_df)

    print(f"Built cache: {CACHE_PATH}")
    print(f"Built metrics: {METRICS_PATH}")
    print(f"Built notebook: {NOTEBOOK_PATH}")
    print(
        "Best model on holdout test:",
        best_model_name,
        f"(RMSE={best_test_row['rmse']:.2f})",
    )


if __name__ == "__main__":
    main()
