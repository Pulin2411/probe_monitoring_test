
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# ========= CONFIG =========
CSV_FILE = "combined_output1.csv"
TIMESTAMP_COL = "StartTime"
TARGET_COL = "AverageLatency"
FORECAST_STEPS = 288
RESAMPLE_FREQ = "5T"
OUTPUT_DIR = "outputs"
# ==========================

def load_and_clean_data(filepath, timestamp_col):
    df = pd.read_csv(filepath, parse_dates=[timestamp_col])
    df.drop_duplicates(subset=timestamp_col, inplace=True)
    df = df.dropna(subset=[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df = df.sort_index()
    return df

def downsample(df, freq):
    return df.resample(freq).mean().interpolate(method="linear")

def run_eda(df, target_col):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing Values Heatmap")
    plt.savefig(f"{OUTPUT_DIR}/missing_heatmap.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    df[target_col].plot()
    plt.title(f"{target_col} Over Time")
    plt.savefig(f"{OUTPUT_DIR}/timeseries_plot.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df[target_col].dropna(), kde=True)
    plt.title(f"{target_col} Distribution")
    plt.savefig(f"{OUTPUT_DIR}/distribution.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()

    df.describe().to_csv(f"{OUTPUT_DIR}/summary_stats.csv")

def train_test_split(series, steps):
    return series[:-steps], series[-steps:]

def forecast_model(train_series, steps, seasonal_period):
    arima_model = auto_arima(train_series, seasonal=False, trace=False, suppress_warnings=True)
    sarima_model = auto_arima(train_series, seasonal=True, m=seasonal_period, trace=False, suppress_warnings=True)

    arima_pred = pd.Series(arima_model.predict(n_periods=steps), index=train_series.index[-steps:])
    sarima_pred = pd.Series(sarima_model.predict(n_periods=steps), index=train_series.index[-steps:])
    return arima_model, sarima_model, arima_pred, sarima_pred

def plot_forecasts(train, test, arima_pred, sarima_pred):
    plt.figure(figsize=(12, 6))
    train.plot(label="Train")
    test.plot(label="Actual", color='black')
    arima_pred.plot(label="ARIMA Forecast", linestyle="--")
    sarima_pred.plot(label="SARIMA Forecast", linestyle="--")
    plt.title("Forecast vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/forecast_comparison.png")
    plt.close()

def main():
    df = load_and_clean_data(CSV_FILE, TIMESTAMP_COL)
    df_downsampled = downsample(df, RESAMPLE_FREQ)

    run_eda(df_downsampled, TARGET_COL)

    series = df_downsampled[TARGET_COL].dropna()
    seasonal_period = 288

    train, test = train_test_split(series, FORECAST_STEPS)

    _, _, arima_pred, sarima_pred = forecast_model(train, FORECAST_STEPS, seasonal_period)

    forecast_df = pd.DataFrame({
        "Actual": test,
        "ARIMA": arima_pred,
        "SARIMA": sarima_pred
    })
    forecast_df.to_csv(f"{OUTPUT_DIR}/predictions.csv")

    plot_forecasts(train, test, arima_pred, sarima_pred)
    print(f"✅ All results saved to ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
