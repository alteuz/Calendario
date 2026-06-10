import os
import pandas as pd
from typing import Optional

class WeatherDataError(Exception):
    """Base exception class for weather data access issues."""
    pass

class WeatherRepository:
    """Manages raw weather records storage and loading."""

    def __init__(self, file_path: str) -> None:
        self._file_path: str = file_path

    def load_weather_records(self) -> pd.DataFrame:
        """Loads and returns the weather dataset from CSV."""
        if not os.path.exists(self._file_path):
            raise WeatherDataError(f"Weather data file not found at: {self._file_path}")
        try:
            weather_records: pd.DataFrame = pd.read_csv(self._file_path)
            return weather_records
        except Exception as error:
            # Re-raise with high-level description to prevent technical leak to UI
            raise WeatherDataError("Error loading weather records from CSV source.") from error
