import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------
# 🌦️ Estilo Temático Climático — UI Mejorada
# ------------------------------------------------------------
st.set_page_config(page_title="Clima Seattle", page_icon="🌤️", layout="wide")

# Fondo temático
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(180deg, #d7ecff 0%, #f7fbff 100%);
    }
    .main > div {
        padding-top: 0 !important;
    }
    .weather-card {
        border-radius: 16px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        backdrop-filter: blur(6px);
        transition: 0.3s;
    }
    .weather-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    }
    h1, h2, h3, h4 {
        color: #2a4d69;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 1️⃣ Cargar datos
# ------------------------------------------------------------
st.title("🌤️ Predicción del Clima en Seattle — Estilo Temático Climático")

with st.container():
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    st.subheader("📥 Carga de datos")
    data = pd.read_csv("seattle_weather_clean.csv")
    st.write("Datos cargados exitosamente.")
    st.markdown('</div>', unsafe_allow_html=True)

# Convertir fechas
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['month'] = data['date'].dt.month

data['season'] = data['month'].apply(lambda x:
                                     'Winter' if x in [12, 1, 2] else
                                     'Spring' if x in [3, 4, 5] else
                                     'Summer' if x in [6, 7, 8] else
                                     'Fall')

# ------------------------------------------------------------
# 2️⃣ One-Hot Encoding
# ------------------------------------------------------------
data = pd.get_dummies(data, columns=['season'], drop_first=False)

# ------------------------------------------------------------
# 3️⃣ Definir variables
# ------------------------------------------------------------
X = data.drop(columns=['weather', 'date'])
y = data['weather']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# ------------------------------------------------------------
# 4️⃣ Entrenar modelo
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5️⃣ Evaluación del modelo
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("📈 Rendimiento del Modelo")
st.metric("Exactitud", f"{accuracy_score(y_test, model.predict(X_test)):.2f}")
st.text(classification_report(y_test, model.predict(X_test)))
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 6️⃣ Tabla resumen mensual
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("📊 Estadísticas climáticas mensuales")
data['month'] = data['month'].astype(int)
monthly_stats = data.groupby('month')[['precipitation', 'temp_max', 'temp_min', 'wind']].median().reset_index()
st.dataframe(monthly_stats)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 7️⃣ Distribución del clima
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("🌤️ Distribución del clima observado")
st.bar_chart(data['weather'].value_counts())
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 8️⃣ Predicción del mes
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("📅 Predicción día por día del mes seleccionado")
mes = st.selectbox("Selecciona un mes:", range(1, 13))

if st.button("🔮 Generar predicción"):

    fechas = pd.date_range(f"2025-{mes:02d}-01", periods=30, freq='D')
    pred_df = pd.DataFrame({"date": fechas})

    valores = monthly_stats[monthly_stats['month'] == mes].iloc[0]

    pred_df['precipitation'] = valores['precipitation'] * np.random.uniform(0.85, 1.15, size=30)
    pred_df['temp_max'] = valores['temp_max'] * np.random.uniform(0.90, 1.10, size=30)
    pred_df['temp_min'] = valores['temp_min'] * np.random.uniform(0.90, 1.10, size=30)
    pred_df['wind'] = valores['wind'] * np.random.uniform(0.85, 1.15, size=30)

    pred_df['month'] = pred_df['date'].dt.month
    pred_df['season'] = pred_df['month'].apply(lambda x:
                                               'Winter' if x in [12, 1, 2] else
                                               'Spring' if x in [3, 4, 5] else
                                               'Summer' if x in [6, 7, 8] else
                                               'Fall')

    pred_df = pd.get_dummies(pred_df, columns=['season'], drop_first=False)

    for col in X.columns:
        if col not in pred_df:
            pred_df[col] = 0

    pred_df_model = pred_df[X.columns]
    pred_imputed = imputer.transform(pred_df_model)
    pronostico = model.predict(pred_imputed)

    pred_df_final = pred_df_model.copy()
    pred_df_final['date'] = fechas
    pred_df_final['Predicted Weather'] = pronostico

    st.dataframe(pred_df_final[['date', 'Predicted Weather']])
    st.bar_chart(pred_df_final['Predicted Weather'].value_counts())

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 9️⃣ Calendario anual estilizado
# ------------------------------------------------------------
st.subheader("🗓️ Calendario anual — Temperatura promedio")

# temp_mean robusto
if 'temp_avg' in data.columns:
    tmp = data[['month', 'temp_avg']].rename(columns={'temp_avg': 'temp_mean'})
else:
    tmp = data[['month', 'temp_max', 'temp_min']]
    tmp['temp_mean'] = (tmp['temp_max'] + tmp['temp_min']) / 2
    tmp = tmp[['month', 'temp_mean']]

calendario = tmp.groupby('month', as_index=False)['temp_mean'].mean()
all_months = pd.DataFrame({'month': list(range(1, 13))})
calendario = all_months.merge(calendario, on='month', how='left')
calendario['temp_mean'] = calendario['temp_mean'].fillna(tmp['temp_mean'].mean())

month_names = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
               "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
calendario['month_name'] = calendario['month'].apply(lambda m: month_names[m - 1])

rows = [calendario.iloc[i:i+3] for i in range(0, 12, 3)]

for row_df in rows:
    cols = st.columns(3)
    for col, (_, row) in zip(cols, row_df.iterrows()):
        temp_val = float(row['temp_mean'])
        color = "#4dabf7" if temp_val < 10 else "#ffa94d" if temp_val < 20 else "#ff6b6b"

        with col:
            st.markdown(f"""
            <div class="weather-card" style="border-left: 6px solid {color}; text-align:center;">
                <h4>{row['month_name']}</h4>
                <p style="font-size: 24px; color:{color}; font-weight: bold;">{temp_val:.1f}°C</p>
                <p>Temperatura promedio</p>
            </div>
            """, unsafe_allow_html=True)
=======
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------
# 🌦️ Estilo Temático Climático — UI Mejorada
# ------------------------------------------------------------
st.set_page_config(page_title="Clima Seattle", page_icon="🌤️", layout="wide")

# Fondo temático
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(180deg, #d7ecff 0%, #f7fbff 100%);
    }
    .main > div {
        padding-top: 0 !important;
    }
    .weather-card {
        border-radius: 16px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        backdrop-filter: blur(6px);
        transition: 0.3s;
    }
    .weather-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    }
    h1, h2, h3, h4 {
        color: #2a4d69;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 1️⃣ Cargar datos
# ------------------------------------------------------------
st.title("🌤️ Predicción del Clima en Seattle — Estilo Temático Climático")

with st.container():
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    st.subheader("📥 Carga de datos")
    data = pd.read_csv("seattle_weather_clean.csv")
    st.write("Datos cargados exitosamente.")
    st.markdown('</div>', unsafe_allow_html=True)

# Convertir fechas
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['month'] = data['date'].dt.month

data['season'] = data['month'].apply(lambda x:
                                     'Winter' if x in [12, 1, 2] else
                                     'Spring' if x in [3, 4, 5] else
                                     'Summer' if x in [6, 7, 8] else
                                     'Fall')

# ------------------------------------------------------------
# 2️⃣ One-Hot Encoding
# ------------------------------------------------------------
data = pd.get_dummies(data, columns=['season'], drop_first=False)

# ------------------------------------------------------------
# 3️⃣ Definir variables
# ------------------------------------------------------------
X = data.drop(columns=['weather', 'date'])
y = data['weather']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# ------------------------------------------------------------
# 4️⃣ Entrenar modelo
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5️⃣ Evaluación del modelo
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("📈 Rendimiento del Modelo")
st.metric("Exactitud", f"{accuracy_score(y_test, model.predict(X_test)):.2f}")
st.text(classification_report(y_test, model.predict(X_test)))
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 6️⃣ Tabla resumen mensual
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("📊 Estadísticas climáticas mensuales")
data['month'] = data['month'].astype(int)
monthly_stats = data.groupby('month')[['precipitation', 'temp_max', 'temp_min', 'wind']].median().reset_index()
st.dataframe(monthly_stats)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 7️⃣ Distribución del clima
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("🌤️ Distribución del clima observado")
st.bar_chart(data['weather'].value_counts())
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 8️⃣ Predicción del mes
# ------------------------------------------------------------
st.markdown('<div class="weather-card">', unsafe_allow_html=True)
st.subheader("📅 Predicción día por día del mes seleccionado")
mes = st.selectbox("Selecciona un mes:", range(1, 13))

if st.button("🔮 Generar predicción"):

    fechas = pd.date_range(f"2025-{mes:02d}-01", periods=30, freq='D')
    pred_df = pd.DataFrame({"date": fechas})

    valores = monthly_stats[monthly_stats['month'] == mes].iloc[0]

    pred_df['precipitation'] = valores['precipitation'] * np.random.uniform(0.85, 1.15, size=30)
    pred_df['temp_max'] = valores['temp_max'] * np.random.uniform(0.90, 1.10, size=30)
    pred_df['temp_min'] = valores['temp_min'] * np.random.uniform(0.90, 1.10, size=30)
    pred_df['wind'] = valores['wind'] * np.random.uniform(0.85, 1.15, size=30)

    pred_df['month'] = pred_df['date'].dt.month
    pred_df['season'] = pred_df['month'].apply(lambda x:
                                               'Winter' if x in [12, 1, 2] else
                                               'Spring' if x in [3, 4, 5] else
                                               'Summer' if x in [6, 7, 8] else
                                               'Fall')

    pred_df = pd.get_dummies(pred_df, columns=['season'], drop_first=False)

    for col in X.columns:
        if col not in pred_df:
            pred_df[col] = 0

    pred_df_model = pred_df[X.columns]
    pred_imputed = imputer.transform(pred_df_model)
    pronostico = model.predict(pred_imputed)

    pred_df_final = pred_df_model.copy()
    pred_df_final['date'] = fechas
    pred_df_final['Predicted Weather'] = pronostico

    st.dataframe(pred_df_final[['date', 'Predicted Weather']])
    st.bar_chart(pred_df_final['Predicted Weather'].value_counts())

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 9️⃣ Calendario anual estilizado
# ------------------------------------------------------------
st.subheader("🗓️ Calendario anual — Temperatura promedio")

# temp_mean robusto
if 'temp_avg' in data.columns:
    tmp = data[['month', 'temp_avg']].rename(columns={'temp_avg': 'temp_mean'})
else:
    tmp = data[['month', 'temp_max', 'temp_min']]
    tmp['temp_mean'] = (tmp['temp_max'] + tmp['temp_min']) / 2
    tmp = tmp[['month', 'temp_mean']]

calendario = tmp.groupby('month', as_index=False)['temp_mean'].mean()
all_months = pd.DataFrame({'month': list(range(1, 13))})
calendario = all_months.merge(calendario, on='month', how='left')
calendario['temp_mean'] = calendario['temp_mean'].fillna(tmp['temp_mean'].mean())

month_names = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
               "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
calendario['month_name'] = calendario['month'].apply(lambda m: month_names[m - 1])

rows = [calendario.iloc[i:i+3] for i in range(0, 12, 3)]

for row_df in rows:
    cols = st.columns(3)
    for col, (_, row) in zip(cols, row_df.iterrows()):
        temp_val = float(row['temp_mean'])
        color = "#4dabf7" if temp_val < 10 else "#ffa94d" if temp_val < 20 else "#ff6b6b"

        with col:
            st.markdown(f"""
            <div class="weather-card" style="border-left: 6px solid {color}; text-align:center;">
                <h4>{row['month_name']}</h4>
                <p style="font-size: 24px; color:{color}; font-weight: bold;">{temp_val:.1f}°C</p>
                <p>Temperatura promedio</p>
            </div>
            """, unsafe_allow_html=True)
