import numpy as np
import pandas as pd
import calendar
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.services.ml_service import WeatherClassifierAdapter

def calculate_season(month: int) -> str:
    """Determines the season based on the month number."""
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"

def preprocess_historical_records(weather_records: pd.DataFrame) -> pd.DataFrame:
    """Cleans and structures the historical weather records."""
    df_copy = weather_records.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy['month'] = df_copy['date'].dt.month.fillna(1).astype(int)
    df_copy['day'] = df_copy['date'].dt.day.fillna(1).astype(int)
    if 'temp_avg' not in df_copy.columns:
        df_copy['temp_avg'] = (df_copy['temp_max'] + df_copy['temp_min']) / 2
    df_copy['season'] = df_copy['month'].apply(calculate_season)
    df_encoded = pd.get_dummies(df_copy, columns=['season'], drop_first=False)
    for season_col in ['season_Winter', 'season_Spring', 'season_Summer', 'season_Fall']:
        if season_col not in df_encoded.columns:
            df_encoded[season_col] = 0
    return df_encoded

def calculate_monthly_medians(weather_records: pd.DataFrame) -> pd.DataFrame:
    """Computes median values for monthly weather metrics."""
    month_int = weather_records['month'].astype(int)
    metrics = ['precipitation', 'temp_max', 'temp_min', 'wind']
    monthly_stats = weather_records.groupby(month_int)[metrics].median().reset_index()
    return monthly_stats

def calculate_monthly_temperatures(weather_records: pd.DataFrame) -> pd.DataFrame:
    """Calculates average monthly temperatures for the annual calendar."""
    df_copy = weather_records.copy()
    df_copy['temp_mean'] = df_copy['temp_avg'] if 'temp_avg' in df_copy.columns else (df_copy['temp_max'] + df_copy['temp_min']) / 2
    temp_by_month = df_copy.groupby('month', as_index=False)['temp_mean'].mean()
    all_months = pd.DataFrame({'month': list(range(1, 13))})
    merged_stats = all_months.merge(temp_by_month, on='month', how='left')
    merged_stats['temp_mean'] = merged_stats['temp_mean'].fillna(df_copy['temp_mean'].mean())
    month_names = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                   "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    merged_stats['month_name'] = merged_stats['month'].apply(lambda m: month_names[m - 1])
    return merged_stats

def get_days_in_month(year: int, month: int) -> int:
    """Returns the number of days in a given month and year."""
    return calendar.monthrange(year, month)[1]

def generate_forecast_features(median_stats: pd.Series, dates: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    """Generates the feature matrix for prediction with controlled perturbation."""
    size = len(dates)
    forecast_df = pd.DataFrame({"date": dates})
    forecast_df['precip_imp'] = forecast_df['precipitation'] = median_stats['precipitation'] * np.random.uniform(0.85, 1.15, size=size)
    forecast_df['temp_max'] = median_stats['temp_max'] * np.random.uniform(0.90, 1.10, size=size)
    forecast_df['temp_min'] = median_stats['temp_min'] * np.random.uniform(0.90, 1.10, size=size)
    forecast_df['wind'] = median_stats['wind'] * np.random.uniform(0.85, 1.15, size=size)
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['day'] = forecast_df['date'].dt.day
    forecast_df['year'] = forecast_df['date'].dt.year
    forecast_df['temp_avg_imp'] = forecast_df['temp_avg'] = (forecast_df['temp_max'] + forecast_df['temp_min']) / 2
    forecast_df['temp_avg_7d'] = forecast_df['temp_avg'].rolling(7, min_periods=1).mean()
    forecast_df['precip_7d'] = forecast_df['precipitation'].rolling(7, min_periods=1).sum()
    seasons = forecast_df['month'].apply(calculate_season)
    for s in ['Fall', 'Spring', 'Summer', 'Winter']:
        forecast_df[f'season_{s}'] = (seasons == s).astype(int)
    for col in columns:
        if col not in forecast_df:
            forecast_df[col] = 0
    return forecast_df[columns]

def train_and_evaluate_model(adapter: WeatherClassifierAdapter, weather_records: pd.DataFrame) -> Tuple[float, str]:
    """Trains the classifier and returns accuracy score and classification report."""
    feature_matrix = weather_records.drop(columns=['weather', 'date'])
    target_series = weather_records['weather']
    train_features, test_features, train_targets, test_targets = train_test_split(
        feature_matrix, target_series, test_size=0.2, random_state=42
    )
    adapter.train_classifier(train_features, train_targets)
    predictions = adapter.predict_weather_labels(test_features)
    accuracy = float(accuracy_score(test_targets, predictions))
    report = str(classification_report(test_targets, predictions))
    return accuracy, report
