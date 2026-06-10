import streamlit as st
import pandas as pd
from typing import Tuple
from src.data.weather_repository import WeatherRepository, WeatherDataError
from src.services.ml_service import WeatherClassifierAdapter, ModelExecutionError
from src.logic.weather_forecaster import (
    preprocess_historical_records,
    calculate_monthly_medians,
    calculate_monthly_temperatures,
    get_days_in_month,
    generate_forecast_features,
    train_and_evaluate_model
)

# ------------------------------------------------------------
# 🌦️ Theme & Styling - Tokenized UI
# ------------------------------------------------------------
st.set_page_config(page_title="Clima Seattle", page_icon="🌤️", layout="wide")

# Theme color tokens
PRIMARY_COLOR = "#2a4d69"
BG_GRADIENT = "linear-gradient(180deg, #d7ecff 0%, #f7fbff 100%)"
CARD_BG = "rgba(255, 255, 255, 0.8)"
CARD_SHADOW = "0 4px 12px rgba(0,0,0,0.1)"
CARD_SHADOW_HOVER = "0 6px 18px rgba(0,0,0,0.15)"

st.markdown(
    f"""
    <style>
    body {{
        background: {BG_GRADIENT};
    }}
    .main > div {{
        padding-top: 0 !important;
    }}
    .weather-card {{
        border-radius: 16px;
        padding: 20px;
        background: {CARD_BG};
        box-shadow: {CARD_SHADOW};
        backdrop-filter: blur(6px);
        transition: 0.3s;
        margin-bottom: 20px;
    }}
    .weather-card:hover {{
        transform: translateY(-4px);
        box-shadow: {CARD_SHADOW_HOVER};
    }}
    h1, h2, h3, h4 {{
        color: {PRIMARY_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌤️ Predicción del Clima en Seattle — Panel de Decisiones")

# ------------------------------------------------------------
# Cache Helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset() -> pd.DataFrame:
    """Loads and preprocesses the weather dataset."""
    repository = WeatherRepository("seattle_weather_clean.csv")
    raw_records = repository.load_weather_records()
    processed_records = preprocess_historical_records(raw_records)
    return processed_records

@st.cache_resource(show_spinner=False)
def get_trained_classifier(processed_records: pd.DataFrame) -> Tuple[WeatherClassifierAdapter, float, str]:
    """Trains the model and returns the adapter, accuracy, and report."""
    adapter = WeatherClassifierAdapter()
    accuracy, report = train_and_evaluate_model(adapter, processed_records)
    return adapter, accuracy, report

# ------------------------------------------------------------
# Application Flow & State Management
# ------------------------------------------------------------
try:
    # 1. Loading State (Loading data and training model)
    with st.spinner("Cargando base de datos climáticos y entrenando modelo predictivo..."):
        weather_records = load_and_preprocess_dataset()
        classifier_adapter, accuracy_score, evaluation_report = get_trained_classifier(weather_records)
    
    # 2. Ideal State: Model Performance
    with st.container():
        st.markdown('<div class="weather-card">', unsafe_allow_html=True)
        st.subheader("📈 Rendimiento del Modelo de Clasificación")
        col_metric, col_details = st.columns([1, 3])
        with col_metric:
            st.metric("Exactitud (Accuracy)", f"{accuracy_score:.2f}")
        with col_details:
            st.text(evaluation_report)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Ideal State: Statistics & Distribution
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="weather-card">', unsafe_allow_html=True)
        st.subheader("📊 Estadísticas climáticas mensuales (Mediana)")
        monthly_medians = calculate_monthly_medians(weather_records)
        st.dataframe(monthly_medians, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="weather-card">', unsafe_allow_html=True)
        st.subheader("🌤️ Distribución de climas observados históricamente")
        climas_counts = weather_records['weather'].value_counts()
        st.bar_chart(climas_counts, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Prediction Panel
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    st.subheader("📅 Predicción día por día del mes seleccionado (Año 2025)")
    selected_month = st.selectbox("Selecciona un mes para pronosticar:", list(range(1, 13)))
    
    # Setup session state for predictions
    if "predictions_df" not in st.session_state:
        st.session_state.predictions_df = None
        st.session_state.predicted_month = None

    generate_button = st.button("🔮 Generar predicción diaria")
    
    if generate_button:
        with st.spinner(f"Simulando clima y generando predicciones para el mes {selected_month}..."):
            # Calculate days in month
            days_count = get_days_in_month(2025, selected_month)
            prediction_dates = pd.date_range(f"2025-{selected_month:02d}-01", periods=days_count, freq='D')
            
            # Fetch median features for that month
            month_medians = monthly_medians[monthly_medians['month'] == selected_month].iloc[0]
            
            # Align features with training columns
            feature_cols = [c for c in weather_records.columns if c not in ['weather', 'date']]
            forecast_features = generate_forecast_features(month_medians, prediction_dates, feature_cols)
            
            # Predict
            predictions = classifier_adapter.predict_weather_labels(forecast_features)
            
            # Store results
            result_df = pd.DataFrame({
                "Fecha": prediction_dates,
                "Pronóstico del Clima": predictions
            })
            st.session_state.predictions_df = result_df
            st.session_state.predicted_month = selected_month

    # 5. Empty State / Ideal Prediction Results State
    if st.session_state.predictions_df is not None and st.session_state.predicted_month == selected_month:
        col_table, col_plot = st.columns([1, 1])
        with col_table:
            st.dataframe(st.session_state.predictions_df, use_container_width=True)
        with col_plot:
            predictions_counts = st.session_state.predictions_df["Pronóstico del Clima"].value_counts()
            st.bar_chart(predictions_counts, use_container_width=True)
    else:
        st.info("💡 Aún no se ha generado la predicción para este mes. Haz clic en 'Generar predicción diaria' para ver el pronóstico.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. Styled Annual Calendar
    st.subheader("🗓️ Calendario anual — Temperatura promedio histórica")
    calendar_stats = calculate_monthly_temperatures(weather_records)
    
    rows = [calendar_stats.iloc[i:i+3] for i in range(0, 12, 3)]
    for row_df in rows:
        cols = st.columns(3)
        for col, (_, row) in zip(cols, row_df.iterrows()):
            temperature_val = float(row['temp_mean'])
            color = "#4dabf7" if temperature_val < 10 else "#ffa94d" if temperature_val < 20 else "#ff6b6b"
            with col:
                st.markdown(f"""
                <div class="weather-card" style="border-left: 6px solid {color}; text-align:center;">
                    <h4>{row['month_name']}</h4>
                    <p style="font-size: 24px; color:{color}; font-weight: bold;">{temperature_val:.1f}°C</p>
                    <p>Temperatura promedio</p>
                </div>
                """, unsafe_allow_html=True)

except WeatherDataError as load_error:
    # 7. Error State
    st.error(f"⚠️ Error al cargar los datos climáticos: {str(load_error)}")
    st.info("Asegúrese de que el archivo 'seattle_weather_clean.csv' esté presente en el mismo directorio del proyecto.")

except ModelExecutionError as execution_error:
    # 7. Error State
    st.error(f"⚠️ Error en el modelo predictivo: {str(execution_error)}")
    st.info("Ocurrió un error al procesar el modelo de aprendizaje automático. Intente recargar la aplicación.")

except Exception as unexpected_error:
    # 7. Error State
    st.error(f"⚠️ Ha ocurrido un error inesperado en la aplicación.")
    st.caption(f"Detalle técnico: {str(unexpected_error)}")
