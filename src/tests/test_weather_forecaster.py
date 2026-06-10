import unittest
import pandas as pd
import numpy as np
from src.logic.weather_forecaster import (
    calculate_season,
    preprocess_historical_records,
    calculate_monthly_medians,
    calculate_monthly_temperatures,
    get_days_in_month,
    generate_forecast_features,
    train_and_evaluate_model
)
from src.services.ml_service import WeatherClassifierAdapter

class TestWeatherForecaster(unittest.TestCase):

    def test_calculate_season(self) -> None:
        self.assertEqual(calculate_season(1), "Winter")
        self.assertEqual(calculate_season(2), "Winter")
        self.assertEqual(calculate_season(12), "Winter")
        self.assertEqual(calculate_season(3), "Spring")
        self.assertEqual(calculate_season(4), "Spring")
        self.assertEqual(calculate_season(5), "Spring")
        self.assertEqual(calculate_season(6), "Summer")
        self.assertEqual(calculate_season(7), "Summer")
        self.assertEqual(calculate_season(8), "Summer")
        self.assertEqual(calculate_season(9), "Fall")
        self.assertEqual(calculate_season(10), "Fall")
        self.assertEqual(calculate_season(11), "Fall")

    def test_get_days_in_month(self) -> None:
        # Non-leap year 2025
        self.assertEqual(get_days_in_month(2025, 2), 28)
        self.assertEqual(get_days_in_month(2025, 1), 31)
        self.assertEqual(get_days_in_month(2025, 4), 30)
        # Leap year 2024
        self.assertEqual(get_days_in_month(2024, 2), 29)

    def test_preprocess_historical_records(self) -> None:
        mock_data = pd.DataFrame({
            "date": ["2012-01-01", "2012-04-01", "2012-07-01", "2012-10-01"],
            "temp_max": [10.0, 15.0, 25.0, 12.0],
            "temp_min": [2.0, 5.0, 15.0, 4.0],
            "precipitation": [5.0, 2.0, 0.0, 8.0],
            "wind": [3.0, 2.5, 1.5, 4.0],
            "weather": ["rain", "drizzle", "sun", "rain"]
        })
        processed = preprocess_historical_records(mock_data)
        
        # Verify columns added
        self.assertIn("month", processed.columns)
        self.assertIn("day", processed.columns)
        self.assertIn("temp_avg", processed.columns)
        self.assertIn("season_Winter", processed.columns)
        self.assertIn("season_Spring", processed.columns)
        self.assertIn("season_Summer", processed.columns)
        self.assertIn("season_Fall", processed.columns)
        
        # Check specific values
        self.assertEqual(processed.loc[0, "month"], 1)
        self.assertEqual(processed.loc[0, "season_Winter"], 1)
        self.assertEqual(processed.loc[0, "season_Spring"], 0)
        self.assertEqual(processed.loc[0, "temp_avg"], 6.0)

    def test_calculate_monthly_medians(self) -> None:
        mock_data = pd.DataFrame({
            "month": [1, 1, 2, 2],
            "precipitation": [2.0, 4.0, 6.0, 8.0],
            "temp_max": [10.0, 12.0, 14.0, 16.0],
            "temp_min": [0.0, 2.0, 4.0, 6.0],
            "wind": [1.0, 3.0, 5.0, 7.0]
        })
        medians = calculate_monthly_medians(mock_data)
        self.assertEqual(len(medians), 2)
        # Check January (month 1) medians
        jan_stats = medians[medians["month"] == 1].iloc[0]
        self.assertEqual(jan_stats["precipitation"], 3.0)
        self.assertEqual(jan_stats["temp_max"], 11.0)

    def test_calculate_monthly_temperatures(self) -> None:
        mock_data = pd.DataFrame({
            "month": [1, 1, 2, 2],
            "temp_avg": [5.0, 7.0, 9.0, 11.0]
        })
        temps = calculate_monthly_temperatures(mock_data)
        self.assertEqual(len(temps), 12)
        jan_row = temps[temps["month"] == 1].iloc[0]
        self.assertEqual(jan_row["temp_mean"], 6.0)
        self.assertEqual(jan_row["month_name"], "Enero")

    def test_generate_forecast_features(self) -> None:
        # 1. Monthly median stats
        medians = pd.Series({
            "month": 1,
            "precipitation": 4.0,
            "temp_max": 10.0,
            "temp_min": 2.0,
            "wind": 3.0
        })
        
        # 2. DatetimeIndex
        dates = pd.date_range("2025-01-01", periods=5, freq='D')
        
        # 3. Target columns
        expected_columns = [
            "precipitation", "temp_max", "temp_min", "wind", "month", "day", "year",
            "temp_avg", "temp_avg_imp", "precip_imp", "temp_avg_7d", "precip_7d",
            "season_Winter", "season_Spring", "season_Summer", "season_Fall"
        ]
        
        forecast = generate_forecast_features(medians, dates, expected_columns)
        
        self.assertEqual(len(forecast), 5)
        self.assertEqual(list(forecast.columns), expected_columns)
        self.assertTrue((forecast["month"] == 1).all())
        self.assertTrue((forecast["year"] == 2025).all())
        self.assertTrue((forecast["season_Winter"] == 1).all())
        # Check perturbation range
        self.assertTrue((forecast["precipitation"] >= 4.0 * 0.85).all())
        self.assertTrue((forecast["precipitation"] <= 4.0 * 1.15).all())

    def test_train_and_evaluate_model(self) -> None:
        # Create a mock dataset with enough rows to train scikit-learn
        # We need a few rows of each class to prevent split/train failures
        mock_data = pd.DataFrame({
            "date": pd.date_range("2012-01-01", periods=20, freq='D'),
            "temp_max": np.random.uniform(5, 25, 20),
            "temp_min": np.random.uniform(0, 15, 20),
            "precipitation": np.random.uniform(0, 10, 20),
            "wind": np.random.uniform(1, 5, 20),
            "weather": ["rain", "sun"] * 10
        })
        processed = preprocess_historical_records(mock_data)
        adapter = WeatherClassifierAdapter()
        
        accuracy, report = train_and_evaluate_model(adapter, processed)
        
        self.assertTrue(adapter.is_trained)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)

if __name__ == "__main__":
    unittest.main()
